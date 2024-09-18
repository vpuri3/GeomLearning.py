#
import torch
import numpy as np

import random

__all__ = [
    "set_seed",
    "num_parameters",
]

def set_seed(seed = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return

def num_parameters(model):
    return sum(p.numel() for p in model.parameters())
#
