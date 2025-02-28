#
import math
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import trunc_normal_, Mlp
from einops import rearrange, repeat

__all__ = [
    "TS1",
]

FastGELU = lambda: nn.GELU(approximate='tanh')

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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, N, C = x.shape

        ### (1) Slice
        # VALUE (B H N C)
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        # KEY (B H N C)
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        # QUERY = weights of self.in_project_slice (dim_head, slice_num)
        #
        # ATTN_WEIGHTS = SLICE_WEIGHTS = softmax(K * Q)
        # SOFTMAX is happening along (-1) query direction.
        # PERCEIVER IO (https://arxiv.org/pdf/2107.14795) did it in (-2) direction
        
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        # V * ATTN_WEIGHTS
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        # IDEAS:
        # - Transolver++: Gimble softmax with temperature = 1 + self.temperature_project(x) (Transolver)
        # - Transolver++: remove fx_mid and use x_mid for both key and value
        # - Make query dependent on x? Is that possible? Would it have the right dimension?
        # - How does this constrast with attention encodings? See Latent diffusion transformer for point cloud generation paper.
        # - Pass in X/Y/Z coordinates to every transolver block and concatenate to the input.

        # - AdaLayerNorm (DiT, https://arxiv.org/abs/2302.07459) like conditioning on t, dt, process paramters
        # - can we accomplish bulk masking with the conditioning idea?
        # - that is make conditioning focus more on top layers
        # - solution: allow the slice weights to be conditioned on t, dt
        
        # - Constant training schedule (lr = 1e-4, weight_decay = 1e-4)
        # - mess with training schedule: OneCycleLR(max_Lr=1e-1, pct_start=0.1), weight_decay=0
        # - Do ReZero (https://proceedings.mlr.press/v161/bachlechner21a/bachlechner21a.pdf)
        #   in Transolver_block

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
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
            slice_num=32,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.att = SliceAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num,
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
        x = x + gate1.unsqueeze(1) * self.att(modulate(self.ln1(x), shift1, scale1))
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
class TS1(nn.Module):
    def __init__(self,
        in_dim,
        out_dim,
        n_layers=5,
        n_hidden=128,
        dropout=0,
        n_head=8,
        mlp_ratio=1,
        slice_num=32,
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
                slice_num=slice_num,
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