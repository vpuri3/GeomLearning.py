#
import torch
import numpy as np
import torch_geometric as pyg
import scipy

__all__ = [
    'interpolate_idw',
]

#======================================================================#
def interpolate_idw(x_src, u_src, x_dst, k=4, pow=2, tol=1e-6):
    tree = scipy.spatial.KDTree(x_src)
    dist, idx = tree.query(x_dst, k=k)

    weight  = 1 / ((dist + tol) ** pow)
    weight /= weight.sum(axis=1, keepdims=True)
    weight  = np.expand_dims(weight, 2)
    u_dst   = np.sum(weight * u_src[idx], axis=1)

    return u_dst

#======================================================================#
#
