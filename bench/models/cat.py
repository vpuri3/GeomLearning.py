#
import math
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import trunc_normal_
from einops import rearrange, einsum

from mlutils.utils import check_package_version_lteq

__all__ = [
    "ClusterAttentionTransformer",
]

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.silu(gates)
    
class GEGLU(nn.Module):
    def forward(self, x):
        if check_package_version_lteq('torch', '2.4.0'):
            kw = {}
        else:
            kw = {'approximate': 'tanh'}
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates, **kw)

# the incremental speedup isn't worth dealing with versioning hell
# FastGELU = nn.GELU
FastGELU = lambda: nn.GELU(approximate='tanh')

ACTIVATIONS = {
    'gelu': FastGELU(),
    'silu': nn.SiLU(),
    'swiglu': SwiGLU(),
    'geglu': GEGLU(),
}

#======================================================================#
# Vanilla Self-Attention Block
#======================================================================#
class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, act: str = None):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = ACTIVATIONS[act] if act else ACTIVATIONS['gelu']
        if act in ['swiglu', 'geglu']:
            self.fc2 = nn.Linear(hidden_dim // 2, out_dim)
        else:
            self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, channel_dim: int, num_heads: int = 8):
        super().__init__()

        assert channel_dim % num_heads == 0, f"channel_dim must be divisible by num_heads. Got {channel_dim} and {num_heads}."

        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads 
        
        self.qkv_proj = nn.Linear(channel_dim, 3 * channel_dim)
        self.out_proj = nn.Linear(channel_dim, channel_dim)

    def forward(self, x):
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        scale = q.shape[-1] ** -0.5
        dots = einsum(q, k, 'b h q d, b h k d -> b h q k') * scale
        attn = F.softmax(dots, dim=-1)
        out = einsum(attn, v, 'b h q k, b h k d -> b h q d')
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)

        return out
    
class SelfAttentionBlock(nn.Module):
    def __init__(self, channel_dim: int, num_heads: int, mlp_ratio: int = 4, act: str = None, if_mlp: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(channel_dim)
        self.mha = MultiHeadedSelfAttention(channel_dim, num_heads)
        self.if_mlp = if_mlp
        if self.if_mlp:
            self.ln2 = nn.LayerNorm(channel_dim)
            self.mlp = MLPBlock(
                in_dim=channel_dim,
                hidden_dim=int(channel_dim * mlp_ratio),
                out_dim=channel_dim,
                act=act,
            )

    def forward(self, x):
        # x: [B, N, C]

        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x)) if self.if_mlp else x

        return x

#======================================================================#
# Cluster Attention
#======================================================================#
class ClusterHeadMixingConv(nn.Module):
    def __init__(self, H:int, M: int):
        super().__init__()
        self.weights = nn.Parameter(torch.empty([H * M, H * M]))
        k = 1 / math.sqrt(H * M)
        nn.init.uniform_(self.weights, -k, k)

    def forward(self, x):

        B, H, M, N = x.shape
        weights = self.weights.view(H * M, H * M, 1)
            
        x = x.view(B, H * M, N)
        x = F.conv1d(x, weights)
        x = x.view(B, H, M, N)

        return x
    
class ClusterAttention(nn.Module):
    def __init__(
            self, channel_dim, num_heads=8, num_clusters=32,
            num_projection_heads=None, num_latent_blocks=1,
            mlp_ratio=2, if_latent_mlp=True, act=None,
            cluster_head_mixing=True,
        ):
        super().__init__()

        self.channel_dim = channel_dim
        self.num_clusters = num_clusters
        self.num_projection_heads = num_projection_heads if num_projection_heads is not None else num_heads
        self.projection_head_dim = self.channel_dim // self.num_projection_heads

        assert self.channel_dim % self.num_projection_heads == 0, f"channel_dim must be divisible by num_projection_heads. Got {self.channel_dim} and {self.num_projection_heads}."

        ### (1) Get cluster weights
        self.cluster_head_mixing  = cluster_head_mixing
        self.k_proj = ResidualMLP(
            in_dim=self.channel_dim, hidden_dim=self.channel_dim, out_dim=self.channel_dim,
            num_layers=2, act=act, input_residual=True, output_residual=True,
        )
        self.v_proj = ResidualMLP(
            in_dim=self.channel_dim, hidden_dim=self.channel_dim, out_dim=self.channel_dim,
            num_layers=2, act=act, input_residual=True, output_residual=True,
        )

        self.latent_q = nn.Parameter(torch.empty(self.channel_dim, self.num_clusters))
        nn.init.normal_(self.latent_q, mean=0.0, std=0.1)
        if self.cluster_head_mixing:
            self.mix = ClusterHeadMixingConv(self.num_projection_heads, self.num_clusters)

        ### (2) Attention among cluster tokens
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(self.channel_dim, num_heads, mlp_ratio=mlp_ratio, act=act, if_mlp=if_latent_mlp)
            for _ in range(num_latent_blocks)
        ])
        
        ### (3) Disaggregate cluster tokens and return
        self.out_proj = nn.Linear(self.channel_dim, self.channel_dim)
        
    def forward(self, x):

        # x: [B N C]

        ### (1) Aggregate cluster tokens

        q = self.latent_q.view(self.num_projection_heads, self.num_clusters, self.projection_head_dim) # [H M D]
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_projection_heads) # [B H N D]
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_projection_heads)

        scores = einsum(q, k, 'h m d, b h n d -> b h m n') # [B H M N]
        if self.cluster_head_mixing:
            scores = self.mix(scores)

        encode_weights = F.softmax(scores, dim=-1) # sum over N
        decode_weights = F.softmax(scores, dim=-2) # sum over M

        z = einsum(encode_weights, v, 'b h m n, b h n d -> b h m d') # [B H M D]
        z = rearrange(z, 'b h m d -> b m (h d)') # [B M C]

        ### (2) Attention among cluster tokens

        for block in self.blocks:
            z = block(z)

        ### (3) Disaggregate cluster tokens

        z = rearrange(z, 'b m (h d) -> b h m d', h=self.num_projection_heads)
        x = einsum(z, decode_weights, 'b h m d, b h m n -> b h n d')
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.out_proj(x)

        return x

#======================================================================#
# BLOCK
#======================================================================#

class ResidualMLP(nn.Module):
    def __init__(
            self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 1,
            act: str = None, input_residual: bool = False, output_residual: bool = False
        ):
        super().__init__()
        if act == 'swiglu':
            print("ResidualMLP does not support swiglu activations. Using silu instead.")
            act = 'silu'
        if act == 'geglu':
            print("ResidualMLP does not support geglu activations. Using gelu instead.")
            act = 'gelu'
        self.act = ACTIVATIONS[act] if act else ACTIVATIONS['gelu']
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fcs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        self.input_residual  = input_residual  and (in_dim  == hidden_dim)
        self.output_residual = output_residual and (hidden_dim == out_dim)

    def forward(self, x):
        x = x + self.act(self.fc1(x)) if self.input_residual else self.act(self.fc1(x))
        for fc in self.fcs:
            x = x + self.act(fc(x))
        x = x + self.fc2(x) if self.output_residual else self.fc2(x)
        return x

class ClusterAttentionBlock(nn.Module):
    def __init__(self,
            num_heads: int,
            channel_dim: int,
            mlp_ratio=2,
            num_clusters=32,
            num_projection_heads=None,
            num_latent_blocks=1,
            if_latent_mlp=True,
            if_pointwise_mlp=True,
            cluster_head_mixing=True,
            act=None,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(channel_dim)
        self.att = ClusterAttention(
            channel_dim,
            num_heads=num_heads,
            num_clusters=num_clusters,
            num_projection_heads=num_projection_heads,
            num_latent_blocks=num_latent_blocks,
            if_latent_mlp=if_latent_mlp,
            mlp_ratio=mlp_ratio,
            act=act,
            cluster_head_mixing=cluster_head_mixing,
        )
        
        self.if_mlp = if_pointwise_mlp
        if self.if_mlp:
            self.ln2 = nn.LayerNorm(channel_dim)
            self.mlp = MLPBlock(
                in_dim=channel_dim,
                hidden_dim=int(channel_dim * mlp_ratio),
                out_dim=channel_dim,
                act=act,
            )
        
    def forward(self, x):
        # x: [B, N, C]

        x = x + self.att(self.ln1(x))
        x = x + self.mlp(self.ln2(x)) if self.if_mlp else x

        return x

class FinalLayer(nn.Module):
    def __init__(self, channel_dim, out_dim, act=None):
        super().__init__()
        self.ln = nn.LayerNorm(channel_dim)
        self.mlp = ResidualMLP(
            in_dim=channel_dim, hidden_dim=channel_dim, out_dim=out_dim,
            num_layers=2, act=act, input_residual=True,
        )

    def forward(self, x):
        x = self.mlp(self.ln(x))
        return x

#======================================================================#
# MODEL
#======================================================================#
class ClusterAttentionTransformer(nn.Module):
    def __init__(self,
        in_dim: int,
        out_dim: int,
        num_layers: int = 5,
        channel_dim: int = 128,
        num_heads: int = 8,
        mlp_ratio: int = 2,
        num_clusters: int = 64,
        num_projection_heads: int = None,
        num_latent_blocks: int = 1,
        if_latent_mlp: bool = True,
        if_pointwise_mlp: bool = True,
        cluster_head_mixing: bool = True,
        act: str = None,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_clusters = num_clusters
        self.num_heads = num_heads

        self.x_proj = ResidualMLP(
            in_dim=in_dim, hidden_dim=channel_dim, out_dim=channel_dim,
            num_layers=2, act=act, output_residual=True,
        )
        
        self.blocks = nn.ModuleList([
            ClusterAttentionBlock(
                num_heads=num_heads,
                channel_dim=channel_dim,
                mlp_ratio=mlp_ratio,
                num_clusters=num_clusters,
                num_projection_heads=num_projection_heads,
                num_latent_blocks=num_latent_blocks,
                if_latent_mlp=if_latent_mlp,
                if_pointwise_mlp=if_pointwise_mlp,
                act=act,
                cluster_head_mixing=cluster_head_mixing,
            )
            for _ in range(num_layers)
        ])

        self.out_proj = FinalLayer(channel_dim, out_dim, act=act)

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0., std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, (nn.LayerNorm,)):
            nn.init.constant_(m.bias, 0.)
            nn.init.constant_(m.weight, 1.)

    def forward(self, x):
        # x: [B, N, C]

        x = self.x_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_proj(x)

        return x

#======================================================================#
#