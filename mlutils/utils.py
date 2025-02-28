#
import torch
from torch import nn
import torch.distributed as dist

import numpy as np

import os
import random

__all__ = [
    "set_seed",

    'set_num_threads',
    "select_device",

    'is_torchrun',
    'dist_backend',
    'dist_setup',
    'dist_finalize',

    "num_parameters",

    "mean_std",
    "shift_scale",
    "normalize",
    "unnormalize",

    "r2",

    "eval_model",
    "eval_gnn",
    "autoregressive_rollout",

    # versioning hell
    'to_numpy',
    'check_package_version_lteq'
]

#=======================================================================#
def to_numpy(t: torch.Tensor):
    '''
    Torch 1.10 compatible equivalent of `t.numpy(force=True)`.
    '''
    return t.detach().cpu().resolve_conj().resolve_neg().numpy()

def check_package_version_lteq(pkg: str, version: str):
    import pkg_resources
    VERSION = pkg_resources.get_distribution(pkg).version
    return pkg_resources.parse_version(VERSION) <= pkg_resources.parse_version(version)

#=======================================================================#
def set_seed(seed = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return

#=======================================================================#
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
    if device is not None:
        return device

    if is_torchrun():
        if not dist.is_initialized():
            dist_setup()
        return int(os.environ['LOCAL_RANK'])

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    if verbose:
        print(f'using device {device}.')

    return device

def dist_backend():
    if dist.is_nccl_available():
        return 'nccl'
    elif dist.is_gloo_available():
        return 'gloo'
    elif dist.is_mpi_available():
        return 'mpi'
    else:
        raise RuntimeError("No suitable backend found!")

def is_torchrun():
    required_env_vars = ['LOCAL_RANK', 'RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
    return all(var in os.environ for var in required_env_vars)

def dist_setup():
    backend = dist_backend()
    if backend != 'nccl':
        print(f'using {backend} backend for torch.distributed.')

    if is_torchrun():
        GLOBAL_RANK = int(os.environ["RANK"])
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        WORLD_SIZE = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(LOCAL_RANK)

        if check_package_version_lteq('torch', '2'): # not sure about version
            dist.init_process_group(
                backend,
                rank=GLOBAL_RANK,
                world_size=WORLD_SIZE,
            )
        else:
            dist.init_process_group(
                backend=backend,
                rank=GLOBAL_RANK,
                world_size=WORLD_SIZE,
                device_id=torch.device(LOCAL_RANK),
            )
    else:
        pass

    return

def dist_finalize():
    if dist.is_initialized():
        dist.destroy_process_group()
    return

#=======================================================================#
def num_parameters(model : nn.Module):
    return sum(p.numel() for p in model.parameters())

def mean_std(x: torch.tensor, channel_dim=-1):
    dims = list(range(x.ndim))
    del dims[channel_dim]
    keepdim = (channel_dim != -1) and (channel_dim != x.ndim-1)

    x_bar = x.mean(dims, keepdim=keepdim)
    x_std = x.std( dims, keepdim=keepdim)

    return x_bar, x_std

def shift_scale(x: torch.tensor, min, max, channel_dim=-1, keepdim=False):
    """
    y = (x - x_min) * (max - min) / (x_max - x_min) + min
    """
    dims = list(range(x.ndim))
    del dims[channel_dim]
    keepdim = (channel_dim != -1) and (channel_dim != x.ndim-1)

    x_min = x_max = x
    for dim in reversed(dims):
        x_min = x_min.min(dim=dim, keepdim=keepdim).values
        x_max = x_max.max(dim=dim, keepdim=keepdim).values

    scale = (x_max - x_min) / (max - min)
    shift = x_min - min * scale
    return shift, scale

def normalize(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x - shift) / scale

def unnormalize(x_norm: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x_norm * scale + shift

#=======================================================================#

def r2(y_pred, y_true):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    y_mean = torch.mean(y_true)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_mean) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()

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
    import torch_geometric as pyg

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
