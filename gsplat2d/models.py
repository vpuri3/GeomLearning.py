import torch
from torch import nn
import torch.nn.functional as F

# local
import mlutils

__all__ = [
    "GSplat",
    "rasterize",
]

class GSplat(nn.Module):
    def __init__(self, N, n=None, image_size=[256, 256, 3]):
        super().__init__()
        self.num_gaussians = N
        if n is None:
            n = N
        H, W, C = image_size

        self.register_buffer("mask", torch.ones(N, dtype=torch.bool))
        self.mask[n:] = False

        self.image_size = image_size
        self.scale = nn.Parameter(torch.logit(torch.rand(N, 2)))
        self.rot   = nn.Parameter(torch.atanh(2 * torch.rand(N, 1) - 1))
        self.mean  = nn.Parameter(torch.rand(N, 2) * 2 - 1)
        self.color = nn.Parameter(torch.rand(N, C))
        self.alpha = nn.Parameter(torch.logit(torch.rand(N, 1)))

    def total_gaussians(self):
        return self.num_gaussians

    def active_gaussians(self):
        return self.mask.sum().item()

    def forward(self):
        mask = self.mask
        scale = torch.sigmoid(self.scale[mask])
        rot   = torch.tanh(self.rot[mask]) * torch.pi / 2
        mean  = torch.tanh(self.mean[mask])
        alpha = torch.sigmoid(self.alpha[mask])
        color = torch.sigmoid(self.color[mask]) * alpha.view(-1, 1)

        return rasterize(scale, rot, mean, color, self.image_size)

    @torch.no_grad()
    def prune(self, mask: torch.Tensor):
        """
        move to end where mask is True
        and zero out its values
        """
        assert mask.dtype == torch.bool # indexable

        if mask.sum().item() > 0:
            return

        for buf in self.buffers():
            buf.data[...] = self._prune(buf.data, mask)
        for param in self.parameters():
            param.data[...] = self._prune(param.data, mask)

        return

    @torch.no_grad()
    def _prune(self, x: torch.Tensor, mask: torch.Tensor):
        assert mask.ndim == 1
        assert len(mask) == len(x)
        assert mask.dtype == torch.bool # indexable
        return torch.cat([x[~mask], 0 * x[mask]], dim=0)

    def split(self, idx: torch.Tensor, scale_factor = 1.6):
        i0, i1 = self.clone(idx)
        self.scale.data[idx  ] /= scale_factor
        self.scale.data[i0:i1] /= scale_factor
        return

    def clone(self, idx):
        if len(idx) == 0:
            return

        # indices of newly formed Gaussians
        i0 = self.active_gaussians()
        i1 = min(i0 + len(idx), self.total_gaussians())

        # ensure new indices correspond to dead gaussians
        assert torch.all(~self.mask[i0:i1])

        # copy over parameters
        self.mask[i0:i1] = True
        for param in self.parameters():
            param.data[i0:i1] = param.data[idx]

        return i0, i1
#

def rasterize(
    scale,    # [N, 2] -- COVARIANCE SCALE
    rot,      # [N, 1] -- COVARIANCE ROTATION ([-pi/2, pi/2])
    mean,     # [N, 2] -- GAUSSIAN LOCATIONS
    color,    # [N, C] -- COLORS
    image_size=[256, 256, 3],
):
    H, W, C = image_size
    N = color.shape[0]
    rot = rot.view(-1)
    device = scale.device

    assert rot.shape[0] == scale.shape[0] == color.shape[0] # num Gaussians
    assert scale.shape[1] == mean.shape[1] == 2               # Dimension == 2
    assert color.shape[1] == C

    #================#
    # COVARIANCE MATRICES
    #================#
    cos_rot = torch.cos(rot) # [N]
    sin_rot = torch.sin(rot)

    S = torch.diag_embed(scale)                   # [N, 2, 2] (diagonal)
    R = torch.stack([                             # [N, 2, 2]
        torch.stack([cos_rot, -sin_rot], dim=-1), # [N, 2]
        torch.stack([sin_rot,  cos_rot], dim=-1), # [N, 2]
    ], dim=-2)

    COV = R @ S @ S @ R.transpose(-1, -2) # [N, 2, 2]
    CINV = torch.inverse(COV)

    #================#
    # CREATE GRID CORRESPONDING TO TARGET IMAGE
    #================#
    L = 5.0
    x = torch.linspace(-L, L, H, device=device)
    y = torch.linspace(-L, L, W, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    xy = torch.stack([xx, yy], dim=-1)         # [H, W, 2]
    xy = xy.unsqueeze(0).expand(N, -1, -1, -1) # [N, H, W, 2]

    #================#
    # FORM GAUSSIAN KERNEL
    #================#
    z = torch.einsum('nxyi, nij, nxyj -> nxy', xy, -0.5 * CINV, xy) # [N, H, W]
    kernel = torch.exp(z) / (2 * torch.pi * torch.sqrt(torch.det(COV))).view(N, 1, 1)

    # normalize kernel
    kernel = kernel / kernel.amax(dim=(-2, -1), keepdim=True) # [N, H, W]

    # add color channels
    kernel = kernel.unsqueeze(1).expand(-1, C, -1, -1) # [N, C, H, W]

    # translate grid
    theta = torch.zeros(N, 2, 3, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, :, 2] = mean

    grid = F.affine_grid(theta, size=(N, C, H, W), align_corners=True)
    kernel = F.grid_sample(kernel, grid, align_corners=True)

    #================#
    # IMAGE
    #================#
    image = kernel * color.unsqueeze(-1).unsqueeze(-1)
    image = image.sum(dim=0)         # sum over Gaussians
    image = torch.clamp(image, 0, 1) # clamp values to [0, 1]
    image = image.permute(1, 2, 0)   # permute to [H, W, C]

    return image
#
