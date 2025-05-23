#
import math
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import trunc_normal_
from einops import rearrange, einsum

__all__ = [
    "SkinnyCAT",
]

#======================================================================#
# activation functions
#======================================================================#

ACTIVATIONS = {
    'gelu': nn.GELU(),
    'silu': nn.SiLU(),
}

#======================================================================#
# MLP Block, Residual MLP Block
#======================================================================#

class ResidualMLP(nn.Module):
    def __init__(
            self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2,
            act: str = None, input_residual: bool = False, output_residual: bool = False
        ):
        super().__init__()
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
    
#======================================================================#
# Cluster Attention
#======================================================================#
class ClusterHeadMixingConv(nn.Module):
    def __init__(self, H:int, M: int):
        super().__init__()
        self.weights = nn.Parameter(torch.empty([H * M, H * M]))
        k = 1 / math.sqrt(H * M)
        nn.init.uniform_(self.weights, -k, k)
        
        # IDEAS:
        # - Low-rank decomposition: W = A @ B.T with A, B ∈ ℝ^[H*M, k]
        # - Group convolutions across heads.
        # - Treat [H, M] as sequence of H tokens of size M or M tokens of size H.
        #   and do attention. Makes mixing weights dynamic.

    def forward(self, x):

        B, H, M, N = x.shape
        weights = self.weights.view(H * M, H * M, 1)

        x = x.view(B, H * M, N)
        x = F.conv1d(x, weights)
        x = x.view(B, H, M, N)

        return x

class ClusterAttention(nn.Module):
    def __init__(self, channel_dim, num_heads=8, num_clusters=32, act=None):
        super().__init__()

        # looks like using ResidualMLP everywhere works best.
        # further, more num_layers work better.
        # consider weight typing bw k_proj, v_proj.

        # num_layers = 2, and CAT block MLP = ResidualMLP(C)
        # mix = True : 1000k - 2.35e-3, 4.02e-3
        # mix = False:  476k - 3.01e-3, 4.37e-3

        self.channel_dim = channel_dim
        self.num_clusters = num_clusters
        self.num_heads = num_heads
        self.head_dim = self.channel_dim // self.num_heads

        assert self.channel_dim % self.num_heads == 0, f"channel_dim must be divisible by num_heads. Got {self.channel_dim} and {self.num_heads}."

        self.latent_q = nn.Parameter(torch.empty(self.channel_dim, self.num_clusters))
        nn.init.normal_(self.latent_q, mean=0.0, std=0.1)

        self.k_proj, self.v_proj = [ResidualMLP(
            in_dim=self.channel_dim, hidden_dim=self.channel_dim, out_dim=self.channel_dim,
            num_layers=2, act=act, input_residual=True, output_residual=True,
        ) for _ in range(2)]

        self.mix = ClusterHeadMixingConv(self.num_heads, self.num_clusters)
        self.out_proj = nn.Linear(self.channel_dim, self.channel_dim)

    def forward(self, x):

        # x: [B N C]

        ### (1) Compute projection weights

        q = self.latent_q.view(self.num_heads, self.num_clusters, self.head_dim) # [H M D]
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads) # [B H N D]
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)

        scores = einsum(q, k, 'h m d, b h n d -> b h m n') # [B H M N]
        scores = self.mix(scores)

        encode_weights = F.softmax(scores, dim=-1) # sum over N
        decode_weights = F.softmax(scores, dim=-2) # sum over M
        
        ### (2) Aggregate cluster tokens

        z = einsum(encode_weights, v, 'b h m n, b h n d -> b h m d') # [B H M D]

        ### (3) Disaggregate cluster tokens

        x = einsum(z, decode_weights, 'b h m d, b h m n -> b h n d')
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.out_proj(x)

        return x

#======================================================================#
# BLOCK
#======================================================================#

class ClusterAttentionBlock(nn.Module):
    def __init__(self,
            channel_dim: int,
            num_heads=8,
            num_clusters=32,
            act=None,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(channel_dim)
        self.ln2 = nn.LayerNorm(channel_dim)
        self.att = ClusterAttention(
            channel_dim, num_heads=num_heads,
            num_clusters=num_clusters, act=act,
        )
        self.mlp = ResidualMLP(
            in_dim=channel_dim, hidden_dim=channel_dim, out_dim=channel_dim,
            num_layers=2, act=act, input_residual=True, output_residual=True,
        )

    def forward(self, x):
        # x: [B, N, C]

        x = x + self.att(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

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
class SkinnyCAT(nn.Module):
    def __init__(self,
        in_dim: int,
        out_dim: int,
        channel_dim: int = 64,
        num_blocks: int = 8,
        num_clusters: int = 32,
        num_heads: int = 8,
        act: str = None,
    ):
        super().__init__()

        self.in_proj = ResidualMLP(
            in_dim=in_dim, hidden_dim=channel_dim, out_dim=channel_dim,
            num_layers=2, act=act, output_residual=True,
        )
        
        self.blocks = nn.ModuleList([
            ClusterAttentionBlock(
                channel_dim=channel_dim,
                num_clusters=num_clusters,
                num_heads=num_heads,
                act=act,
            )
            for _ in range(num_blocks)
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

        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_proj(x)

        return x

#======================================================================#
#