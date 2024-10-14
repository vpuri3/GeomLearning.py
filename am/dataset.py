import torch
import torch_geometric as pyg
import numpy as np

import os

__all__ = [
    'makedata'
]

def elem_to_adj(elem): # [E, 8]
    E = elem.size(0)
    N = elem.max()
    A = np.zeros(2, E)
    return A
