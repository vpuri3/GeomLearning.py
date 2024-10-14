import torch
from torch import nn
from torch.nn import functional as F

# local
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gn = nn.GroupNorm(32, c)
        self.atten = SelfAttention(1, c)

    def forward(self, x):
        B, C, H, W = x.shape
        resid = x
        x = self.gn(x)
        x = x.view((B, C, H * W)).transpose(-1, -2) # [B, H*W, C]
        x = self.atten(x)
        x = x.transpose(-1, -2)
        x = x.view((B, C, H, W))
        return x + resid
#

class VAE_ResidualBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.gn1 = nn.GroupNorm(32, ci)
        self.gn2 = nn.GroupNorm(32, co)
        self.conv1 = nn.Conv2d(ci, co, kernle_size=3, padding=1)
        self.conv2 = nn.Conv2d(co, co, kernel_size=3, padding=1)

        if ci == co:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(ci, co, kernel_size=1, padding=0)

    def forward(self, x):
        resid = x
        x = self.gn1(x)
        x = F.silu(x)
        x = self.conv1(x)

        x = self.gn2(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x + self.residual_layer(resid)
#

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # [B, C, H, W]
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            #  [B, 128, H, W]
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # [B, 128, H/2, W/2]
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),
            # [B, 256, H/2, W/2]
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # [B, 256, H/4, W/4]
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),
            # [B, 512, H/4, W/4]
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            # [B, 512, H/8, W/8]
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            #
            VAE_AttentionBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.Silu(),
            #
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # [B, 8, H/8, W/8]
            nn.Conv2d(8, 8, kernel_size=3, padding=0),
        )

    def forward(self, x, noise):
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # pad with zeros on right and bottom in downsampling block
                # Why? padding and downsamplingn should be asymmetric
                x = F.pad(x, (0, 1, 0, 1)) # (left, right, top, bottom)
            x = module(x)
        #
        mean, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, -30, 20)
        std = logvar.exp().sqrt()
        x = mean + std * noise
        x = x * 0.18215
        return x
#


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # [B, 4, H/8, W/8]
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            # [B, 512, H/8, W/8]
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            #
            nn.Upsample(scale_factor=2),
            # [B, 512, H/4, W/4]
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            #
            nn.Upsample(scale_factor=2),
            # [B, 512, H/2, W/2]
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            #
            nn.Upsample(scale_factor=2),
            # [B, 512, H, W]
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            #
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = x / 0.18215
        for module in self:
            x = module(x)
        return x
#
