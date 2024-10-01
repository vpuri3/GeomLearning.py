#
import torch
from torch import nn

import torch_scatter
import torch_geometric as pyg
from torch_geometric.nn import MessagePassing

import math

__all__ = [
    "MLP",
    "Sine",
    "SDFClamp",
    #
    "C2d_block",
    "CT2d_block",
    #
    "UNet",
    "double_conv",
    #
    "MeshGraphNet",
    "MeshGraphLayer",
]

#------------------------------------------------#
# MLP
#------------------------------------------------#
def MLP(
    in_dim, out_dim, width, hidden_layers, act=None,
    siren=False, w0=10.0,
):
    if siren:
        act = Sine()
    if act is None:
        act = nn.Tanh()

    layers = []
    layers.extend([nn.Linear(in_dim, width), act])
    for _ in range(hidden_layers):
        layers.extend([nn.Linear(width, width), act])
    layers.extend([nn.Linear(width, out_dim)])

    if siren:
        for (i, layer) in enumerate(layers):
            w = w0 if i == 0 else 1.0
            if isinstance(layer, nn.Linear):
                siren_init_(layer, w)

    return nn.Sequential(*layers)

#------------------------------------------------#
# SIREN
# initialization modified from https://github.com/dalmia/siren/
#------------------------------------------------#
class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return torch.sin(x)

@torch.no_grad()
def siren_init_(layer: nn.Linear, w):
    fan = nn.init._calculate_correct_fan(layer.weight, "fan_in")
    bound = math.sqrt(6 / fan)

    layer.bias.uniform_(-math.pi, math.pi)
    layer.weight.uniform_(-bound, bound)
    layer.weight.mul_(w)

    return

#------------------------------------------------#
# SDF Clamp
#------------------------------------------------#
class SDFClamp(nn.Module):
    def __init__(self, eps, act = nn.Tanh()):
        super().__init__()
        self.eps = eps
        self.act = act
    def forward(self, x):
        return self.eps * self.act(x)

#------------------------------------------------#
# Conv blocks
#------------------------------------------------#
def C2d_block(ci, co, k=None, ctype=None, act=None, lnsize=None):
    """
    ctype: "kto1": [N, Ci,  k,  k] --> [N, Co, 1, 1] (kernel_size=k)
    ctype:   "2x": [N, Ci, 2H, 2W] --> [N, Co, H, W] (kernel_size=3 then max pool)
    ctype:   "4x": [N, Ci, 4H, 4W] --> [N, Co, H, W] (kernel_size=7)
    """

    layers = []

    if ctype == "kto1":
        conv = nn.Conv2d(ci, co, kernel_size=k, stride=1, padding=0)
    elif ctype == "2x": 
        conv = nn.Conv2d(ci, co, kernel_size=3, stride=1, padding=1)
    elif ctype == "4x": 
        conv = nn.Conv2d(ci, co, kernel_size=7, stride=4, padding=3)
    else:
        raise NotImplementedError()

    layers.append(conv)

    if lnsize is not None:
        layers.append(nn.LayerNorm(lnsize))

    if act is not None:
        layers.append(act)

    if ctype == "2x":
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        layers.append(pool)

    return nn.Sequential(*layers)

#------------------------------------------------#
# ConvTranspose blocks
#------------------------------------------------#
def CT2d_block(ci, co, k=None, ctype=None, act=None, lnsize=None):
    """
    ctype: "1tok": [N, Ci, 1, 1] --> [N, Co,  k,  k] (kernel_size=k)
    ctype:   "2x": [N, Ci, H, W] --> [N, Co, 2H, 2W] (kernel_size=4)
    ctype:   "4x": [N, Ci, H, W] --> [N, Co, 4H, 4W] (kernel_size=8)
    """
    layers = []

    if ctype == "1tok":
        conv = nn.ConvTranspose2d(ci, co, kernel_size=k, stride=1, padding=0)
    elif ctype == "2x":
        conv = nn.ConvTranspose2d(ci, co, kernel_size=4, stride=2, padding=1)
    elif ctype == "4x":
        conv = nn.ConvTranspose2d(ci, co, kernel_size=8, stride=4, padding=2)
    else:
        raise NotImplementedError()

    layers.append(conv)

    if lnsize is not None:
        layers.append(nn.LayerNorm(lnsize))

    if act is not None:
        layers.append(act)

    return nn.Sequential(*layers)

#------------------------------------------------#
# UNet
# https://github.com/usuyama/pytorch-unet
#------------------------------------------------#
def double_conv(ci, co, k):
    assert k % 2 == 1
    p = (k - 1) // 2

    return nn.Sequential(
        nn.Conv2d(ci, co, kernel_size=k, padding=p),
        nn.ReLU(inplace=True),
        nn.Conv2d(co, co, kernel_size=k, padding=p),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, ci, co, k, w=256, H=None, W=None):
        super().__init__()

        ######################
        self.dconv_down1 = double_conv(ci ,  64, k)
        self.dconv_down2 = double_conv(64 , 128, k)
        self.dconv_down3 = double_conv(128, 256, k)
        self.dconv_down4 = double_conv(256, 512, k)
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dconv_up3 = double_conv(256 + 512, 256, k)
        self.dconv_up2 = double_conv(128 + 256, 128, k)
        self.dconv_up1 = double_conv(128 +  64,  64, k)
        
        self.conv_last = nn.Conv2d(64, co, 1)

        # self.ln_down1 = nn.LayerNorm([ 64, H//2, W//2])
        # self.ln_down2 = nn.LayerNorm([128, H//2, W//2])
        # self.ln_down3 = nn.LayerNorm([256, H//2, W//2])
        # self.ln_down4 = nn.LayerNorm([512, H//2, W//2])
        ######################

        # self.dconv_down1 = double_conv(ci, w, k)
        # self.dconv_down2 = double_conv( w, w, k)
        # self.dconv_down3 = double_conv( w, w, k)
        # self.dconv_down4 = double_conv( w, w, k)
        #
        # self.maxpool = nn.MaxPool2d(2)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #
        # self.dconv_up3 = double_conv(2*w, w, k)
        # self.dconv_up2 = double_conv(2*w, w, k)
        # self.dconv_up1 = double_conv(2*w, w, k)
        #
        # self.conv_last = nn.Conv2d(w, co, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   

        x = self.dconv_down4(x)

        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out

#------------------------------------------------#
# MeshGraphNets
# https://medium.com/stanford-cs224w/learning-mesh-based-flow-simulations-on-graph-networks-44983679cf2d
#------------------------------------------------#
class MeshGraphNet(nn.Module):
    def __init__(self, ci_node, ci_edge, co, hidden_dim, num_layers, **kwargs):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(ci_node, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(ci_edge, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.processor = nn.ModuleList()
        for _ in range(num_layers):
            layer = MeshGraphLayer(hidden_dim, hidden_dim, **kwargs)
            self.processor.append(layer)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, co),
        )

        return

    def forward(self, data):
        xn, xe = data.x, data.edge_attr

        xn = self.node_encoder(xn)
        xe = self.edge_encoder(xe)

        for layer in self.processor:
            xn, xe = layer(xn, xe, data.edge_index)

        xn = self.decoder(xn)

        return xn
#

class MeshGraphLayer(MessagePassing):
    def __init__(self, ci, co, **kwargs):
        super().__init__(**kwargs)

        self.node_mlp = nn.Sequential(
            nn.Linear(2 * ci, co),
            nn.ReLU(),
            nn.Linear(co, co),
            nn.LayerNorm(co),
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * ci, co),
            nn.ReLU(),
            nn.Linear(co, co),
            nn.LayerNorm(co),
        )

        return

    def forward(self, xn, xe, edge_index):
        msg, ye = self.propagate(edge_index, x=xn, edge_attr=xe)

        yn = torch.cat([msg, xn], dim=1) # cat messages with node features
        yn = self.node_mlp(yn)           # apply node MLP
        yn = yn + xn                     # residual connection

        return yn, ye

    def message(self, x_i, x_j, xe):
        ye = torch.cat([x_i, x_j, xe], dim=1) # cat node and edge features
        ye = self.edge_mlp(ye)                # apply edge MLP
        ye = xe + ye                          # residual connection

        return ye

    def aggregate(self, ye, edge_index):
        node_dim=0
        msg = torch_scatter.scatter(ye, edge_index[0,:], dim=node_dim, reduce='sum')

        return msg, ye
#
#------------------------------------------------#
#
