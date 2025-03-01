#
import math
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import trunc_normal_, Mlp
from einops import rearrange, einsum

__all__ = [
    "TS",
]

# # the incremental speedup isn't worth dealing with versioning hell
# FastGELU = lambda: nn.GELU(approximate='tanh')
FastGELU = nn.GELU

#======================================================================#
# SLICE ATTENTION
#======================================================================#
class SliceAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0., num_slices=32):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.head_dim = hidden_dim // num_heads
        self.num_heads = num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        self.temperature_project = nn.Linear(hidden_dim, self.num_heads)

        self.to_kv_slice = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.xq = nn.Parameter(torch.empty(num_heads, num_slices, self.head_dim))
        torch.nn.init.orthogonal_(self.xq)

        self.qkv_proj = nn.Parameter(torch.empty(self.num_heads, self.head_dim, 3 * self.head_dim))
        trunc_normal_(self.qkv_proj, std=0.02)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        # x: [B, N, C]
        
        ### (1) Slicing

        xk, xv = self.to_kv_slice(x).chunk(2, dim=-1)
        xk = rearrange(xk, 'b n (h d) -> b h n d', h=self.num_heads) # [B, H, N, D]
        xv = rearrange(xv, 'b n (h d) -> b h n d', h=self.num_heads)

        temperature = 0.5 + F.softplus(self.temperature_project(x)) # [B, N, H]
        temperature = temperature.transpose(1, 2).unsqueeze(-2)     # [B, H, 1, N]

        slice_scores = einsum(self.xq, xk, 'h m d, b h n d -> b h m n') # [B, H, M, N]
        slice_weights = F.softmax(slice_scores / temperature, dim=-2)
        slice_norm = slice_weights.sum(dim=-1) # [B, H, M]

        slice_token = einsum(slice_weights, xv, 'b h m n, b h n d -> b h m d') # [B, H, M, D]
        slice_token = slice_token / (slice_norm.unsqueeze(-1) + 1e-5)
        
        ### (2) Attention among slice tokens

        qkv_slice_token = einsum(slice_token, self.qkv_proj, 'b h m d, h d e -> b h m e')
        q_slice_token, k_slice_token, v_slice_token = qkv_slice_token.chunk(3, dim=-1)

        dots = einsum(q_slice_token, k_slice_token.transpose(-1, -2), 'b h q d, b h d q -> b h q d') * self.scale
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        out_slice_token = einsum(attn, v_slice_token, 'b h q d, b h q d -> b h q d')

        ### (3) Deslice

        out_x = einsum(out_slice_token, slice_weights, 'b h m d, b h m n -> b h n d')
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')

        return self.to_out(out_x)

#======================================================================#
# BLOCK
#======================================================================#
class Block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            mlp_ratio=4,
            num_slices=32,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.att = SliceAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_slices=num_slices,
        )

        self.mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=int(hidden_dim * mlp_ratio),
            out_features=hidden_dim,
            act_layer=FastGELU,
            drop=dropout,
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
class TS(nn.Module):
    def __init__(self,
        in_dim,
        out_dim,
        n_layers=5,
        n_hidden=128,
        dropout=0,
        n_head=8,
        mlp_ratio=1,
        num_slices=32,
    ):
        super().__init__()
        
        self.x_embedding = Mlp(
            in_dim,
            n_hidden * 2,
            n_hidden,
            act_layer=FastGELU,
            drop=dropout,
        )
        
        self.blocks = nn.ModuleList([
            Block(
                num_heads=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                num_slices=num_slices,
            )
            for _ in range(n_layers)
        ])
        self.final_layer = FinalLayer(n_hidden, out_dim)

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

    def forward(self, data):
        x = data.x.unsqueeze(0) # [B=1, N, C]
        x = self.x_embedding(x) # [B=1, N, C]
        for block in self.blocks:
            x = block(x)
        x = self.final_layer(x)

        return x[0] # [N, C]

#======================================================================#
#