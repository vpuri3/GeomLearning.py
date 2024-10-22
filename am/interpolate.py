#
import torch
import numpy as np
import scipy

# local
from am.utils import get_zmax_list

__all__ = [
    'interpolate_idw',
    'combine_meshes',
    'make_finest_mesh',
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
def combine_meshes(verts1, elems1, verts2, elems2):
    verts = torch.cat([verts1, verts2], dim=0)
    verts, idx = torch.unique(verts, dim=0, return_inverse=True)

    e1 = idx[:len(verts1)][elems1]
    e2 = idx[len(verts1):][elems2]

    elems = torch.cat([e1, e2], dim=0)
    elems = elems.unique(dim=0) # still has overlapping elements

    # rm different permutations of overlapping elements
    elems_sort = elems.sort(dim=1)[0]
    _, idx_unique = torch.unique(elems_sort, dim=0, return_inverse=True)
    idx_unique = torch.unique(idx_unique, sorted=False)

    elems = elems[idx_unique]

    return verts, elems

#======================================================================#
def make_finest_mesh(dataset, outdir, icase, tol=1e-6):
    N = len(dataset)
    zmax = get_zmax_list(dataset)

    V = dataset[0].pos
    E = dataset[0].elems

    for i in range(1,N):
        verts = dataset[i].pos
        elems = dataset[i].elems
        V, E = combine_meshes(V, E, verts, elems)

    return V, E

#======================================================================#
#
