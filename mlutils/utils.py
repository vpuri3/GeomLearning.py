#
import torch
from torch import nn
import torch_geometric as pyg

import numpy as np

import os
import random

__all__ = [
    "set_seed",
    "select_device",
    "num_parameters",
    "mean_std",
    "normalize",
    "unnormalize",
    "eval_model",
    "eval_gnn",
    "autoregressive_rollout",
]

#=======================================================================#
def set_seed(seed = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return

def set_num_threads(threads=None):
    if threads is not None:
        threads = os.cpu_count()

    torch.set_num_threads(threads)

    os.environ["OMP_NUM_THREADS"]        = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"]   = str(threads)
    os.environ["MKL_NUM_THREADS"]        = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"]    = str(threads)

    return

#=======================================================================#
def select_device(device=None, verbose=False):
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    # TODO: check if device is available
    if verbose:
        print(f"using device {device}")

    return device

#=======================================================================#
def num_parameters(model : nn.Module):
    return sum(p.numel() for p in model.parameters())

def mean_std(x: torch.tensor, channel_dim = None):
    dims = list(range(x.ndim))
    if channel_dim is None:
        channel_dim = x.ndim-1
    del dims[channel_dim]

    x_bar = x.mean(dims, keepdim=True)
    x_std = x.std( dims, keepdim=True)

    return x_bar, x_std

def normalize(x: torch.Tensor, x_bar: torch.Tensor, x_std: torch.Tensor):
    return (x - x_bar) / x_std

def unnormalize(x_norm: torch.Tensor, x_bar: torch.Tensor, x_std: torch.Tensor):
    return x_norm * x_std + x_bar

#=======================================================================#
def eval_model(
    x: torch.Tensor, model: nn.Module, device=None,
    batch_size=1, verbose=False,
):
    device = select_device(device, verbose=verbose)
    loader = torch.utils.data.DataLoader(x, shuffle=False, batch_size=batch_size)
    model  = model.to(device)

    ys = []
    for xx in loader:
        xx = xx.to(device)
        yy = model(xx).to("cpu")
        del xx
        ys.append(yy)

    model = model.to("cpu")
    y = torch.cat(ys, dim=0)
    torch.cuda.empty_cache()
    return y

def eval_gnn(
    data, model: nn.Module, device=None,
    batch_size=1, verbose=False,
):
    device = select_device(device, verbose=verbose)
    loader = pyg.loader.DataLoader(data, shuffle=False, batch_size=batch_size)
    model  = model.to(device)

    ys = []
    for batch in loader:
        batch = batch.to(device)
        y = model(batch).to("cpu")
        del batch
        ys.append(y)

    model = model.to("cpu")
    y = torch.cat(ys, dim=0)
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
#=======================================================================#
#
