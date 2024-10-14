import torch
from torch import nn
from torch.nn import functional as F

class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__(self, H, D)
        self.D = D
        self.H = H
        self.D_ = D // H

        assert (D % H) == 0
        self.proj1 = nn.Linear(dim, 3 * dim)
        self.proj2 = nn.Linear(dim, dim)

    def forward(self, x, causal_mask=False):
        B, N, D = x.shape
        q, k, v = self.proj1(x).chunk(3, dim=-1)

        shape = (B, N, H, self.D_)
        q = q.view(shape).transpose(1, 2) # [B, H, N, D_]
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)

        w = q @ k.transpose(-1, -2) # [B, H, N, N]

        if causal_mask:
            # upper triangle (above principal diagonal is 1)
            mask = torch.ones_like(w, dtype=torch.bool).triu(1)
            weight.maksed_fill(mask, -torch.inf)

        w /= math.sqrt(self.D_)
        w = F.softmax(w, dim=-1) # row sum is 1
        out = w @ v # [B, H, N, N] * [B, H, N, D_] -> [B, H, N, D_]

        out = out.transpose(1, 2) # [B, N, H, D_]
        out = out.view(B, N, D)
        out = self.proj2(out)
        return out
#

class CrossAttention(nn.Module):
    def __init__(self, H, De, Dc):
        super().__init__()
        self.H = H
        self.D_ = De // H
        assert (De % H) == 0

        # De: embedding dim
        # Dc: cross-attention dim

        self.q_proj = nn.Linear(De, De) # query     from sequence x
        self.k_proj = nn.Linear(Dc, De) # key/value from sequence y
        self.v_proj = nn.Linear(Dc, De)
        self.out_proj = nn.Linear(De, De)

    def forward(self, x, y):
        # x (latent) : [B, Nq, Dq]
        # y (context): [B, Nk, Dk]
        B, Nx, _ = x.shape
        B, Ny, _ = y.shape

        shape_q = (B, Nx, H, self.D_)
        shape_k = (B, Ny, H, self.D_)
        q = self.q_proj(x).view(shape_q).transpose(1, 2) # [B, H, Nx, D_]
        k = self.k_proj(x).view(shape_k).transpose(1, 2) # [B, H, Ny, D_]
        v = self.v_proj(x).view(shape.k).transpose(1, 2)

        w = q @ k.transpose(-1, -2) # [B, H, Nx, Ny]
        w /= math.sqrt(self.D_)
        w = F.softmax(w, dim=-1)
        out = w @ v # [B, H, Nx, Dq]

        out = out.transpose(1, 2) # [B, H, Nx, Dq]
        out = out.view(B, N, self.De)
        out = self.out_proj(out)

        return out
#
