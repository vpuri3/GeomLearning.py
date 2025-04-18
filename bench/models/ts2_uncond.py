#
import math
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import trunc_normal_
from einops import rearrange, einsum

from mlutils.utils import check_package_version_lteq

__all__ = [
    "TS2Uncond",
]

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.silu(gates)
    
class GEGLU(nn.Module):
    def __init__(self):
        super().__init__()
        if check_package_version_lteq('torch', '2.4.0'):
            self.kw = {}
        else:
            self.kw = {'approximate': 'tanh'}

    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates, **self.kw)

# the incremental speedup isn't worth dealing with versioning hell
# FastGELU = nn.GELU
FastGELU = lambda: nn.GELU(approximate='tanh')

ACTIVATIONS = {
    'gelu': FastGELU(),
    'silu': nn.SiLU(),
    'swiglu': SwiGLU(),
    'geglu': GEGLU(),
}

def mha(q, k, v):
    scale = q.shape[-1] ** -0.5
    dots = einsum(q, k, 'b h q d, b h k d -> b h q k') * scale
    attn = F.softmax(dots, dim=-1)
    out = einsum(attn, v, 'b h q k, b h k d -> b h q d')
    return out, attn

class SliceHeadMixingConv(nn.Module):
    def __init__(self, H:int, M: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=H * M, out_channels=H * M, kernel_size=1, bias=False)

    def forward(self, x):
        B, H, M, N = x.shape
        
        x = x.view(B, H * M, N)
        x = self.conv(x)
        x = x.view(B, H, M, N)

        return x

#======================================================================#
# SLICE ATTENTION
#======================================================================#
class SliceAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, num_slices=32, qk_norm=False):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_slices = num_slices
        self.head_dim = hidden_dim // num_heads
        self.qk_norm = qk_norm

        ### (1) Get slice weights
        self.wt_kv_proj = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        self.wtq = nn.Parameter(torch.empty(self.num_heads, self.num_slices, self.head_dim))
        nn.init.normal_(self.wtq, mean=0.0, std=0.1)

        self.mix = SliceHeadMixingConv(self.num_heads, self.num_slices)
        self.ln = nn.LayerNorm(self.head_dim)

        ### (2) Attention among slice tokens
        self.qkv_proj = nn.Linear(self.hidden_dim, 3 * self.hidden_dim, bias=False)
        
        ### (3) Deslice and return
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        
    def forward(self, x):
        
        # x: [B N C]

        ### (1) Slicing

        xq = self.wtq # [H M D]
        xk, xv = self.wt_kv_proj(x).chunk(2, dim=-1)
        xk = rearrange(xk, 'b n (h d) -> b h n d', h=self.num_heads) # [B H N D]
        xv = rearrange(xv, 'b n (h d) -> b h n d', h=self.num_heads)

        if self.qk_norm:
            xq = F.normalize(xq, dim=-1)
            xk = F.normalize(xk, dim=-1)
        
        slice_scores = einsum(xq, xk, 'h m d, b h n d -> b h m n') # [B H M N]
        slice_scores = self.mix(slice_scores)
        slice_weights = F.softmax(slice_scores, dim=-2)
        
        slice_token = einsum(slice_weights, xv, 'b h m n, b h n d -> b h m d') # [B H M D]
        slice_token = slice_token / (slice_weights.sum(dim=-1, keepdim=True) + 1e-5)
        slice_token = self.ln(slice_token)

        ### (2) Attention among slice tokens

        slice_token = rearrange(slice_token, 'b h m d -> b m (h d)')
        qkv_token = self.qkv_proj(slice_token)
        q_token, k_token, v_token = qkv_token.chunk(3, dim=-1)
        q_token = rearrange(q_token, 'b m (h d) -> b h m d', h=self.num_heads)
        k_token = rearrange(k_token, 'b m (h d) -> b h m d', h=self.num_heads)
        v_token = rearrange(v_token, 'b m (h d) -> b h m d', h=self.num_heads)
        out_token, _ = mha(q_token, k_token, v_token) # [B H M D]
        
        # mix attn weights in mha
        # residual connectino and layer_norm (gated with alpha?)
        
        ### (3) Deslice

        out = einsum(out_token, slice_weights, 'b h m d, b h m n -> b h n d')
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

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            mlp_ratio=2,
            num_slices=32,
            qk_norm=False,
            act=None,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.att = SliceAttention(
            hidden_dim,
            num_heads=num_heads,
            num_slices=num_slices,
            qk_norm=qk_norm,
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
class TS2Uncond(nn.Module):
    def __init__(self,
        in_dim,
        out_dim,
        num_layers=5,
        hidden_dim=128,
        num_heads=8,
        mlp_ratio=1,
        num_slices=32,
        qk_norm=False,
        act=None,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_slices = num_slices
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
                num_slices=num_slices,
                qk_norm=qk_norm,
                act=act,
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