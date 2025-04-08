#
import math
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import trunc_normal_, Mlp
from einops import rearrange, einsum

__all__ = [
    "TS4",
]

# # the incremental speedup isn't worth dealing with versioning hell
# FastGELU = lambda: nn.GELU(approximate='tanh')
FastGELU = nn.GELU

def mha(q, k, v):
    scale = q.shape[-1] ** -0.5
    dots = einsum(q, k, 'b h q d, b h k d -> b h q k') * scale
    attn = F.softmax(dots, dim=-1)
    out = einsum(attn, v, 'b h q k, b h k d -> b h q d')
    return out, attn

#======================================================================#
# Embeddings
#======================================================================#
class TimeEmbedding(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    
    Ref: https://github.com/facebookresearch/DiT/blob/main/models.py
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.

        Ref: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

#======================================================================#
# SLICE ATTENTION
#======================================================================#
class SliceAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0., num_slices=32, qk_norm=False):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_slices = num_slices
        self.head_dim = hidden_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.qk_norm = qk_norm

        # (1) Get slice weights
        self.wt_kv_proj = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        self.wt_q_proj = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_dim, self.num_heads * self.head_dim * self.num_slices),)
        self.alpha = nn.Parameter(torch.ones([self.num_heads]))

        # (2) Attention among slice tokens
        self.qkv_proj = nn.Parameter(torch.empty(self.num_heads, self.head_dim, 3 * self.head_dim))
        trunc_normal_(self.qkv_proj, std=0.02)

        # (3) Desclice and return
        self.to_out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, c):

        # x: [B, N, C], c: [B, C]
        
        ### (1) Slicing

        xq = self.wt_q_proj(c)
        xk, xv = self.wt_kv_proj(x).chunk(2, dim=-1)
        xq = rearrange(xq, 'b (h m d) -> b h m d', m=self.num_slices, h=self.num_heads) # [B, H, M, D]
        xk = rearrange(xk, 'b n (h d) -> b h n d', h=self.num_heads) # [B, H, N, D]
        xv = rearrange(xv, 'b n (h d) -> b h n d', h=self.num_heads)

        if self.qk_norm:
            xq = F.normalize(xq, dim=-1)
            xk = F.normalize(xk, dim=-1)

        alpha = self.alpha.view(-1, 1, 1)
        slice_scores = einsum(xq, xk, 'b h m d, b h n d -> b h m n') # [b, h, m, n]
        slice_weights = F.softmax(slice_scores * alpha, dim=-2)
        slice_norm = slice_weights.sum(dim=-1) # [B, H, M]
        slice_token = einsum(slice_weights, xv, 'b h m n, b h n d -> b h m d') # [B, H, M, D]
        slice_token = slice_token / (slice_norm.unsqueeze(-1) + 1e-5)
        
        ### (2) Attention among slice tokens

        qkv_token = einsum(slice_token, self.qkv_proj, 'b h m d, h d e -> b h m e')

        q_token, k_token, v_token = qkv_token.chunk(3, dim=-1)
        out_token, attn_weights = mha(q_token, k_token, v_token)

        ### (3) Deslice

        out = einsum(out_token, slice_weights, 'b h m d, b h m n -> b h n d')
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        return out

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
            qk_norm=False,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.att = SliceAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_slices=num_slices,
            qk_norm=qk_norm,
        )

        self.mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=int(hidden_dim * mlp_ratio),
            out_features=hidden_dim,
            act_layer=FastGELU,
            drop=dropout,
        )

    def forward(self, x, c):
        # x: [B, N, C]
        # c: [B, C]
        x = x + self.att(self.ln1(x), c)
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
class TS4(nn.Module):
    def __init__(self,
        in_dim,
        out_dim,
        n_layers=5,
        n_hidden=128,
        dropout=0,
        n_head=8,
        mlp_ratio=1,
        num_slices=32,
        qk_norm=False,
    ):
        super().__init__()
        
        self.x_embedding = Mlp(
            in_dim,
            n_hidden * 2,
            n_hidden,
            act_layer=FastGELU,
            drop=dropout,
        )
        
        self.t_embedding = TimeEmbedding(n_hidden)
        self.d_embedding = TimeEmbedding(n_hidden)
        
        self.blocks = nn.ModuleList([
            Block(
                num_heads=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                num_slices=num_slices,
                qk_norm=qk_norm,
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
        x = data.x.unsqueeze(0) # space dim [B=1, N, C]

        assert x.ndim == 3, "x must be [N, C], that is batch size must be 1"

        if data.get('t_val', None) is not None:
            t = data.t_val.item()
        elif data.get('t', None) is not None:
            t = data.t[0].item()
        else:
            raise RuntimeError(f't_val, t is None in {data}')
        
        t = torch.tensor([t], dtype=x.dtype, device=x.device) # [B=1]
        t = self.t_embedding(t)
        c = t # [B=1, C]
        
        if data.get('dt_val', None) is not None:
            d = data.dt_val.item()
            d = torch.tensor([d], dtype=x.dtype, device=x.device) # [B=1]
            d = self.d_embedding(d)
            c = c + d
        
        x = self.x_embedding(x) # [B=1, N, C]
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x)

        return x[0] # [N, C]

#======================================================================#
#