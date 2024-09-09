#
import torch
from torch import nn

# class MLP(nn.Module):
#     def __init__(self,):
#         pass
# #

def MLP(i, o, w, h, act = "tanh"):
    activation = None
    if act == "tanh":
        activation = nn.Tanh()
    elif act == "relu":
        activation = nn.ReLU()
    elif act == "sigmoid":
        activation = nn.Sigmoid()

    ii = nn.Linear(i, w)
    hh = nn.Sequential(nn.Linear(w, w), activation)
    hd = [hh for _ in range(h)]
    fn = nn.Linear(w, o)

    return nn.Sequential(
        ii,
        *hd,
        fn,
    )
#
