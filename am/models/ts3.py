#
import math
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import trunc_normal_, Mlp
from einops import rearrange, einsum

__all__ = [
    "TS3",
]

# # the incremental speedup isn't worth dealing with versioning hell
# FastGELU = lambda: nn.GELU(approximate='tanh')
FastGELU = nn.GELU

def modulate(x, shift, scale):
    '''
    Modulate the input x with shift and scale with
    AdaLayerNorm (DiT, https://arxiv.org/abs/2302.07459)
    '''
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

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
    def __init__(self, hidden_dim, num_heads=8, dropout=0., num_slices=32):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.head_dim = hidden_dim // num_heads
        self.num_heads = num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        self.temperature_project = nn.Linear(hidden_dim, self.num_heads)

        self.to_kv_slice = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.xq = nn.Parameter(torch.empty(1, num_heads, num_slices, self.head_dim))
        torch.nn.init.orthogonal_(self.xq)

        # TODO: compare with standard MHA
        self.qkv_proj = nn.Parameter(torch.empty(self.num_heads, self.head_dim, 3 * self.head_dim))
        trunc_normal_(self.qkv_proj, std=0.02)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, c):

        # x: [B, N, C]
        # c: [B, C]
        
        # TODO: make slice weights dependent on c via query

        # IDEAS:
        # - Transolver++: Gimble softmax with temperature = 1 + self.temperature_project(x) (Transolver)
        # - Transolver++: remove xv and use xk for both key and value

        ### (1) Slicing

        xk, xv = self.to_kv_slice(x).chunk(2, dim=-1)
        xk = rearrange(xk, 'b n (h d) -> b h n d', h=self.num_heads) # [B, H, N, D]
        xv = rearrange(xv, 'b n (h d) -> b h n d', h=self.num_heads)

        temperature = 0.5 + F.softplus(self.temperature_project(x)) # [B, N, H]
        temperature = temperature.transpose(1, 2).unsqueeze(-2)     # [B, H, 1, N]

        slice_scores = einsum(self.xq, xk, 'b h m d, b h n d -> b h m n') # [B, H, M, N]
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
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    def forward(self, x, c):
        # x: [B, N, C]
        # c: [B, C]
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate1.unsqueeze(1) * self.att(modulate(self.ln1(x), shift1, scale1), c)
        x = x + gate2.unsqueeze(1) * self.mlp(modulate(self.ln2(x), shift2, scale2))
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Linear(hidden_dim, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.ln(x), shift, scale)
        x = self.mlp(x)
        return x

#======================================================================#
# MODEL
#======================================================================#
class TS3(nn.Module):
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
        
        self.t_embedding = TimeEmbedding(n_hidden)
        self.d_embedding = TimeEmbedding(n_hidden)
        
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
        x = data.x.unsqueeze(0) # space dim [B=1, N, C]

        assert x.ndim == 3, "x must be [N, C], that is batch size must be 1"
        if data.get('t_val', None) is None or data.get('dt_val', None) is None:
            raise RuntimeError(f't or d is None in {data}')
        
        t = data.t_val.item()
        t = torch.tensor([t], dtype=x.dtype, device=x.device) # [B=1]
        t = self.t_embedding(t)
        
        d = data.dt_val.item()
        d = torch.tensor([d], dtype=x.dtype, device=x.device) # [B=1]
        d = self.d_embedding(d)
        
        c = t + d               # [B=1, C]
        x = self.x_embedding(x) # [B=1, N, C]
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)

        return x[0] # [N, C]

#======================================================================#
#