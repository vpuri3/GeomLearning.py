#
import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np

import random

__all__ = [
    "set_seed",
    "select_device",
    "num_parameters",
    "eval_model",
    "autoregressive_rollout",
]

def set_seed(seed = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return

def select_device(device=None, verbose=False):
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    if verbose:
        print(f"using device {device}")

    return device

def num_parameters(model : nn.Module):
    return sum(p.numel() for p in model.parameters())

def eval_model(
    x : torch.Tensor,
    model : nn.Module,
    device=None,
    batch_size=1,
    verbose=False,
):

    device = select_device(device, verbose=verbose)

    model = model.to(device)
    loader = DataLoader(x, shuffle=False, batch_size=batch_size)

    ys = []
    for xx in loader:
        xx = xx.to(device)
        yy = model(xx).to("cpu")

        del xx
        ys.append(yy)

    model = model.to("cpu")
    y = torch.cat(ys, dim=0)

    # clear GPU memory
    torch.cuda.empty_cache()

    return y

def autoregressive_rollout(
    x : torch.Tensor,
    model : nn.Module,
    num_iters,
    process=None, # (y0, y1) --> y_next_input, y_save
    save=None,    # [ys] --> y
    device=None,
    verbose=False,
):
    if process is None:
        process = lambda y0, y1 : y1
    if save is None:
        save = lambda ys : torch.stack(ys, dim=0)

    device = select_device(device, verbose=verbose)

    ys = [x]
    y0 = x.to(device)
    model = model.to(device)

    for iter in range(num_iters):
        y1 = model(y0)
        y2 = process(y0, y1)

        ys.append(y2.to("cpu"))
        y0 = y2
    #

    model = model.to("cpu")
    y = save(ys)

    # clear GPU memory
    torch.cuda.empty_cache()

    return y
#
