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
FastGELU = nn.GELU
# FastGELU = lambda: nn.GELU(approximate='tanh')

ACTIVATIONS = {
    'gelu': FastGELU(),
    'silu': nn.SiLU(),
    'swiglu': SwiGLU(),
    'geglu': GEGLU(),
}

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads 
        
        # Attn scores [B H M M] can be mixed whichever way.
        # self.mix = ClusterHeadMixingConv(self.num_heads, self.num_clusters)
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        scale = q.shape[-1] ** -0.5
        dots = einsum(q, k, 'b h q d, b h k d -> b h q k') * scale
        # dots = self.mix(dots) # remove scale?
        attn = F.softmax(dots, dim=-1)
        out = einsum(attn, v, 'b h q k, b h k d -> b h q d')
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)

        return out

class ClusterHeadMixingConv(nn.Module):
    def __init__(self, H:int, M: int, positive_weights=False):
        super().__init__()
        self.positive_weights = positive_weights
        self.weights = nn.Parameter(torch.empty([H * M, H * M]))
        if self.positive_weights:
            nn.init.normal_(self.weights, mean=0., std=1.)
        else:
            k = 1 / math.sqrt(H * M)
            nn.init.uniform_(self.weights, -k, k)

    def forward(self, x):
        B, H, M, N = x.shape
        
        weights = self.weights.view(H * M, H * M, 1)
        if self.positive_weights:
            weights = F.softmax(weights, dim=1)
            
        x = x.view(B, H * M, N)
        x = F.conv1d(x, weights)
        x = x.view(B, H, M, N)

        return x
    
#======================================================================#
# CLUSTER ATTENTION
#======================================================================#
class ClusterAttention(nn.Module):
    def __init__(
            self, hidden_dim, num_heads=8, num_clusters=32, qk_norm=False,
            conv2d=False, H=None, W=None,
        ):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_clusters = num_clusters
        self.head_dim = hidden_dim // num_heads
        self.qk_norm = qk_norm

        self.conv2d = conv2d
        self.H = H
        self.W = W

        ### (1) Get cluster weights
        if self.conv2d:
            self.wt_kv_proj = nn.Conv2d(self.hidden_dim, 2 * self.hidden_dim, kernel_size=3, padding=1)
        else:
            self.wt_kv_proj = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)

        self.wtq = nn.Parameter(torch.empty(self.num_heads, self.num_clusters, self.head_dim))
        nn.init.normal_(self.wtq, mean=0.0, std=0.1)

        self.mix = ClusterHeadMixingConv(self.num_heads, self.num_clusters)
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        ### (2) Attention among cluster tokens
        self.mha = MultiHeadAttention(self.hidden_dim, self.num_heads)
        self.alpha = nn.Parameter(torch.full([self.hidden_dim], 1.0))
        
        ### (3) Disaggregate cluster tokens and return
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        
    def forward(self, x):
        
        # x: [B N C]

        ### (1) Aggregate cluster tokens

        q = self.wtq # [H M D]
        
        if self.conv2d:
            x = rearrange(x, 'b (l w) c -> b c l w', l=self.H)
            k, v = self.wt_kv_proj(x).chunk(2, dim=1) # [B C H W]
            k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=self.num_heads) # [B H N D]
            v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=self.num_heads)
        else:
            k, v = self.wt_kv_proj(x).chunk(2, dim=-1) # [B N C]
            k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads) # [B H N D]
            v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        if self.qk_norm:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
        
        scores = einsum(q, k, 'h m d, b h n d -> b h m n') # [B H M N]
        scores = self.mix(scores)
        
        encode_weights = F.softmax(scores, dim=-1)
        decode_weights = F.softmax(scores, dim=-2)

        z = einsum(encode_weights, v, 'b h m n, b h n d -> b h m d') # [B H M D]
        z = rearrange(z, 'b h m d -> b m (h d)')

        ### (2) Attention among cluster tokens
        
        z = self.ln1(z)
        z = z * self.alpha + self.mha(z)
        z = self.ln2(z)
        z = rearrange(z, 'b m (h d) -> b h m d', h=self.num_heads)

        ### (3) Disaggregate cluster tokens

        out = einsum(z, decode_weights, 'b h m d, b h m n -> b h n d')
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)
        
        return out

#======================================================================#
# BLOCK
#======================================================================#

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act=None):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = ACTIVATIONS[act] if act else ACTIVATIONS['gelu']
        if act in ['swiglu', 'geglu']:
            self.fc2 = nn.Linear(hidden_features // 2, out_features)
        else:
            self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    """Transformer encoder block."""

    def __init__(self,
            num_heads: int,
            hidden_dim: int,
            mlp_ratio=2,
            num_clusters=32,
            qk_norm=False,
            act=None,
            conv2d=False,
            H=None,
            W=None,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.att = ClusterAttention(
            hidden_dim,
            num_heads=num_heads,
            num_clusters=num_clusters,
            qk_norm=qk_norm,
            conv2d=conv2d,
            H=H,
            W=W,
        )
        
        self.mlp = MLP(
            in_features=hidden_dim,
            hidden_features=int(hidden_dim * mlp_ratio),
            out_features=hidden_dim,
            act=act,
        )
        
    def forward(self, x):
        # x: [B, N, C]
       
        x = x + self.att(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.mlp(self.ln(x))
        return x

#======================================================================#
# MODEL
#======================================================================#
class ClusterAttentionTransformer(nn.Module):
    def __init__(self,
        in_dim,
        out_dim,
        num_layers=5,
        hidden_dim=128,
        num_heads=8,
        mlp_ratio=1,
        num_clusters=32,
        qk_norm=False,
        act=None,
        conv2d=False,
        H=None,
        W=None,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_clusters = num_clusters
        self.num_heads = num_heads

        self.x_embedding = MLP(
            in_features=in_dim,
            hidden_features=hidden_dim * 2,
            out_features=hidden_dim,
            act=act,
        )
        
        self.blocks = nn.ModuleList([
            Block(
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                mlp_ratio=mlp_ratio,
                num_clusters=num_clusters,
                qk_norm=qk_norm,
                act=act,
                conv2d=conv2d,
                H=H,
                W=W,
            )
            for _ in range(num_layers)
        ])

        self.final_layer = FinalLayer(hidden_dim, out_dim)

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm,)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        # x: [B, N, C]

        x = self.x_embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_layer(x)

        return x

#======================================================================#
#