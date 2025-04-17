#
import math
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import trunc_normal_
from einops import rearrange, einsum

from mlutils.utils import check_package_version_lteq

__all__ = [
    "TS3Uncond",
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

def sinkhorn(scores, n_iters=5, row_first=True, do_exp=True):
    """
    Perform Sinkhorn normalization to make a matrix doubly stochastic.
    Assumes input scores is (..., M, N).
    """
    for _ in range(n_iters):
        if row_first:
            scores = scores - torch.logsumexp(scores, dim=-1, keepdim=True)  # Row
            scores = scores - torch.logsumexp(scores, dim=-2, keepdim=True)  # Column
        else:
            scores = scores - torch.logsumexp(scores, dim=-2, keepdim=True)  # Column
            scores = scores - torch.logsumexp(scores, dim=-1, keepdim=True)  # Row
    if do_exp:
        return torch.exp(scores)
    else:
        return scores
    
class SliceHeadMixingConv(nn.Module):
    def __init__(self, H:int, M: int, positive_weights=False):
        super().__init__()
        self.positive_weights = positive_weights
        # self.weights = nn.Parameter(torch.empty([H * M, H * M]))
        # self.bias = nn.Parameter(torch.zeros([H * M]))
        # if self.positive_weights:
        #     self.weights.data.zero_()
        # else:
        #     trunc_normal_(self.weights, std=0.02)
        self.conv = nn.Conv1d(in_channels=H * M, out_channels=H * M, kernel_size=1, bias=False)

    def forward(self, x):
        B, H, M, N = x.shape
        
        # weights = self.weights.view(H * M, H * M, 1)
        # if self.positive_weights:
        #     weights = F.softmax(weights, dim=1)
            
        x = x.view(B, H * M, N)
        # x = F.conv1d(x, weights, self.bias)
        x = self.conv(x)
        x = x.view(B, H, M, N)

        return x

#======================================================================#
# SLICE ATTENTION
#======================================================================#
class SliceAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, num_slices=32, qk_norm=False, k_val=None):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_slices = num_slices
        self.head_dim = hidden_dim // num_heads
        self.qk_norm = qk_norm
        self.k_val = k_val

        ### (1) Get slice weights
        self.wt_kv_proj = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        self.wtq = nn.Parameter(torch.empty(self.num_heads, self.num_slices, self.head_dim))
        trunc_normal_(self.wtq, std=0.02)

        self.alpha1 = nn.Parameter(torch.full([self.num_heads],  1.))
        self.alpha2 = nn.Parameter(torch.full([self.num_heads], -4.))

        self.mix1 = SliceHeadMixingConv(self.num_heads, self.num_slices)
        self.mix2 = SliceHeadMixingConv(self.num_heads, self.num_slices, positive_weights=True)

        self.ln1 = nn.LayerNorm(self.head_dim)
        self.ln2 = nn.LayerNorm(self.head_dim)

        ### (2) Attention among slice tokens
        self.qkv_proj = nn.Linear(self.hidden_dim, 3 * self.hidden_dim, bias=False)
        
        ### (3) Deslice and return
        self.to_out = nn.Linear(self.hidden_dim, self.hidden_dim)
        
    def forward(self, x, gamma:float=None, noise=None):
        
        # x: [B N C]

        ### (1) Slicing

        xq = self.wtq # [H M D]
        xk, xv = self.wt_kv_proj(x).chunk(2, dim=-1)
        xk = rearrange(xk, 'b n (h d) -> b h n d', h=self.num_heads) # [B H N D]
        xv = rearrange(xv, 'b n (h d) -> b h n d', h=self.num_heads)

        if self.qk_norm:
            xq = F.normalize(xq, dim=-1)
            xk = F.normalize(xk, dim=-1)
        
        alpha1 = self.alpha1.view(-1, 1, 1)
        alpha2 = F.sigmoid(self.alpha2).view(-1, 1, 1)
        slice_scores = einsum(xq, xk, 'h m d, b h n d -> b h m n') # [B H M N]
        slice_scores = self.mix1(slice_scores)
        slice_weights1 = F.softmax(slice_scores * alpha1, dim=-2)
        # slice_weights2 = self.mix2(slice_weights1)
        # slice_weights = slice_weights1 + alpha2 * slice_weights2
        slice_weights = slice_weights1
        
        # normalize weights to have unit sum in dim=-1, dim=-2

        # self.mn_mix2.weight.data.exp_()
        # slice_weights = slice_weights / (slice_weights.sum(dim=-2, keepdim=True) + 1e-5)

        slice_token = einsum(slice_weights, xv, 'b h m n, b h n d -> b h m d') # [B H M D]
        slice_token = slice_token / (slice_weights.sum(dim=-1, keepdim=True) + 1e-5)
        
        # activation/ out_proj/ LayerNorm here?
        slice_token = self.ln1(slice_token) # normalize heads independently

        ### (2) Attention among slice tokens
        slice_token = rearrange(slice_token, 'b h m d -> b m (h d)')
        qkv_token = self.qkv_proj(slice_token)
        q_token, k_token, v_token = qkv_token.chunk(3, dim=-1)
        q_token = rearrange(q_token, 'b m (h d) -> b h m d', h=self.num_heads)
        k_token = rearrange(k_token, 'b m (h d) -> b h m d', h=self.num_heads)
        v_token = rearrange(v_token, 'b m (h d) -> b h m d', h=self.num_heads)
        out_token, attn_weights = mha(q_token, k_token, v_token) # [B H M D]
        
        # out_token = self.ln2(out_token)
        
        ### (3) Deslice

        out = einsum(out_token, slice_weights, 'b h m d, b h m n -> b h n d')
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        return out, slice_weights, alpha1, torch.zeros(self.num_heads, self.num_slices), attn_weights

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
            k_val=None,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.att = SliceAttention(
            hidden_dim,
            num_heads=num_heads,
            num_slices=num_slices,
            qk_norm=qk_norm,
            k_val=k_val,
        )
        
        self.mlp = MLP(
            in_features=hidden_dim,
            hidden_features=int(hidden_dim * mlp_ratio),
            out_features=hidden_dim,
            act=act,
        )
        
    def forward(self, x, gamma:float=None, noise=None):
        # x: [B, N, C]
       
        _x, *stats = self.att(self.ln1(x), gamma=gamma, noise=noise)
        x = x + _x
        x = x + self.mlp(self.ln2(x))
        return x, *stats

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
class TS3Uncond(nn.Module):
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
        k_val=None,
    ):
        super().__init__()
        
        self.k_val = k_val
        self.gamma = 0.0

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
                k_val=k_val,
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
            
    def forward(self, x, gamma:float=None, noise=None, return_stats: bool=False):
        # x: [B, N, C]
        x = self.x_embedding(x) # [B, N, C]
        
        if (gamma is not None) and self.training:
            self.gamma = gamma

        if return_stats:
            slice_weights = []
            temperature = []
            bias = []
            attn_weights = []

        for block in self.blocks:
            x, *stats = block(x, gamma=gamma, noise=noise)
            if return_stats:
                slice_weights.append(stats[0])
                temperature.append(stats[1])
                bias.append(stats[2])
                attn_weights.append(stats[3])

        x = self.final_layer(x)

        if return_stats:
            return x, slice_weights, temperature, bias, attn_weights
        else:
            return x

#======================================================================#
#