#
import torch
from torch import nn

import math

__all__ = [
    "MLP",
    "Sine",
    #
    "SDFClamp",
    #
    "C2d_block",
    #
    "CT2d_block",
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

def siren_init_(layer: nn.Linear, w):
    with torch.no_grad():

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
# Conv
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
# ConvTranspose
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
#
