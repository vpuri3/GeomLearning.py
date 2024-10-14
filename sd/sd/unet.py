import torch
from torch import nn
from torch.nn import functional as F

from attention import SelfAttention CrossAttention

class Upsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x
#

class _Sequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif: isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)

        return x
#

class UNET_AttentionBlock(nn.Module):
    def __init__(self, H, De, Dc=768):
        super().__init__()
        C = H * De
        self.gn = nn.GruopNorm(32, C, eps=1e-6)

        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)
        self.ln3 = nn.LayerNorm(C)

        self.atten1 = SelfAttention(H, C)
        self.atten2 = CrossAttention(H, C, Dc)

        self.conv1 = nn.Conv2d(C, C, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(C, C, kernel_size=1, padding=0)

        self.lin1 = nn.Linear(C, 4*C*2)
        self.lin2 = nn.Linear(4*C, C)

    def forward(self, x, context):
        # x: [B, C, H, W]
        # context: [B, N, Dc]
        B, C, H, W = x.shape

        resid1 = x
        x = self.gn(x)
        x = self.conv1(x)

        # self attention block
        x = x.view((B, C, H * W)).transpose(-1, -2) # [B, H*W, C]
        resid2 = x
        x = self.ln1(x)
        x = self.atten1(x)
        x += resid2

        # cross attention block
        resid2 = x
        x = self.ln2(x)
        x = self.atten2(x, context)
        x += resid2

        # Normalization + FF with GeGLU activation and skip connection
        resid2 = x
        x = self.ln3(x)
        x, gate = self.lin1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.lin2(x)
        x += resid2

        x = x.transpose(-1, -2).view(B, C, H, W)
        return x + resid1
#

class UNET_ResidualBlock(nn.Module):
    def __init__(self, ci, co, n_time=1280):
        self.gn1 = nn.GroupNorm(32, ci)
        self.gn2 = nn.GroupNorm(32, co)
        self.lin = nn.Linear(n_time, co)
        self.conv1 = nn.Conv2d(ci, co, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(co, co, kernel_size=3, padding=1)

        if ci == co:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(ci, co, kernel_size=1, padding=0)

    def forward(self, x, time):
        resid = x
        x = self.gn1(x)
        x = F.silu(x)
        x = self.conv1(x)

        time = F.silu(time)
        time = self.lin(time)

        x = x + time.view(1, -1, 1, 1)
        x = self.gn2(x)
        x = F.silu(x)
        x = self.conv2(x)

        return x + self.residual_layer(resid)
#

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        conv_kw = {"kernel_size" : , "padding" : 1}

        self.encoder = nn.ModuleList([
            # [B, 4, H/8, W/8]
            _Sequential(nn.Conv2d(4, 320, **conv_kw)), # [B, 320, H/8, W/8]
            _Sequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            _Sequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            #
            _Sequential(nn.Conv2d(320, 320, **conv_kw, stride=2)), # [B, 320, H/16, W/16]
            _Sequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            _Sequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            #
            _Sequential(nn.Conv2d(640, 640, **conv_kw, stride=2)), # [B, 640, H/32, W/32]
            _Sequential(UNET_ResidualBlock( 640, 1280), UNET_AttentionBlock(8, 160)),
            _Sequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            #
            _Sequential(nn.Conv2d(1280, 1280, **conv_kw, stride=2)), # [B, 1280, H/64, W/64]
            _Sequential(UNET_ResidualBlock(1280, 1280)),
            _Sequential(UNET_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280),
        )

        self.decoder = nn.ModuleList([
            #
            _Sequential(UNET_ResidualBlock(2560, 1280)), 
            _Sequential(UNET_ResidualBlock(2560, 1280)), 
            _Sequential(UNET_ResidualBlock(2560, 1280), UpSample(1280)), 
            #
            _Sequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            _Sequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            _Sequential(UNET_ResidualBlock(1920, 1280), UpSample(1280)), 
            #
            _Sequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            _Sequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            _Sequential(UNET_ResidualBlock( 960, 640), UpSample(640)), 
            #
            _Sequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            _Sequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 80)),
            _Sequential(UNET_ResidualBlock(640, 320), UpSample(320)), 
        ])

    def forward(self, x):
        skip = []
        for layer in self.encoder:
            x = layer(x)
            skip.append(x)

        x = self.bottleneck(x)

        for layer in self.decoder:
            x = torch.cat([x, skip.pop()], dim=1)
            x = layer(x)

        return x
