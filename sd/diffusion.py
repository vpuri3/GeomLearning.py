
import torch
from torch import nn
from torch.nn import functional as F

from unet import UNET

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.lin1 = nn.Linear(n_embd, 4 * n_embd)
        self.lin2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x): # [..., n_embd] -> [..., 4 * n_embd]
        x = self.lin1(x)
        x = F.silu(x)
        x = self.lin2(x)
        return x

class OutputLayer(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.gn = nn.GroupNorm(32, ci)
        self.conv = nn.Conv2d(ci, co, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.gn(x)
        x = F.silu(x)
        x = self.conv(x)
        return x

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = OutputLayer(320, 4)

    def forward(self, latent, context, time):
        # latent : noised up latent image [B, 4, H/8, W/8] 
        # context: text embedding         [B, N, D]
        # time   : diffusion time-step    [1, 320]

        time = self.time_embedding(time)       # [1, 320 * 4]
        out = self.unet(latent, context, time) # [B, 320, H/8, W/8]
        out = self.final(out)                  # [B, 4, H/8, H/8]
        return out
