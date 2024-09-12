#
import torch
from torch import nn

import math

__all__ = [
    "MLP",
    "Sine",
    "SDFClamp",
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
#
