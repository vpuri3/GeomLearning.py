#
import torch
import torch_geometric as pyg
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
def interpolate_idw(x_src, u_src, x_dst, k=4, pow=2, tol=1e-6, workers=-1, tree=None):
    if tree is None:
        tree = scipy.spatial.KDTree(x_src)
    dist, idx = tree.query(x_dst, k=k, workers=workers)

    weight  = 1 / ((dist + tol) ** pow)
    weight /= weight.sum(axis=1, keepdims=True)
    weight  = np.expand_dims(weight, 2)
    u_dst   = np.sum(weight * u_src[idx], axis=1)

    return u_dst

#======================================================================#
def bounding_box(verts, elems):
    hex_verts = verts[elems]             # [E, 8, 3]
    min, _ = torch.min(hex_verts, dim=1) # [E, 3]
    max, _ = torch.max(hex_verts, dim=1)
    return min, max

def is_contained(min1, max1, min2, max2):
    """ checks if element 1 is contained in element 2 """
    return torch.all(min1 >= min2, dim=-1) * torch.all(max1 <= max2, dim=-1)

def rm_overlapping_elems(V, E):
    mins, maxs = bounding_box(V, E)

    # O(N^2) check
    contained = is_contained(
        mins.unsqueeze(1), maxs.unsqueeze(1), # i
        mins.unsqueeze(0), maxs.unsqueeze(0), # j
    )
    contained.diagonal().mul_(False)

    ij = torch.argwhere(contained)
    idx_rm = torch.unique(ij[:,1])

    idx_keep = [i for i in range(E.shape[0]) if i not in idx_rm]
    elems_refined = E[idx_keep]

    return elems_refined

#======================================================================#
def combine_meshes(V1, E1, V2, E2, rm_overlap=True):
    """
    does not account for overlapping elemenets
    """
    V = torch.cat([V1, V2], dim=0)
    V, idx = torch.unique(V, dim=0, return_inverse=True)

    e1 = idx[:len(V1)][E1]
    e2 = idx[len(V1):][E2]

    E = torch.cat([e1, e2], dim=0)
    E = E.unique(dim=0)

    # rm redundant elements
    e_sorted = E.sort(dim=1)[0]
    idx_uniq = torch.unique(e_sorted, dim=0, return_inverse=True)[1]
    idx_uniq = torch.unique(idx_uniq, sorted=False)

    E = E[idx_uniq]

    if rm_overlap:
        E = rm_overlapping_elems(V, E)

    return V, E

def make_finest_mesh(dataset, tol=1e-6):
    N = len(dataset)
    zmax = get_zmax_list(dataset)

    V = dataset[0].pos
    E = dataset[0].elems

    # combine all meshes
    for i in range(1,N):
        verts = dataset[i].pos
        elems = dataset[i].elems
        V, E = combine_meshes(V, E, verts, elems, rm_overlap=False)

    # rm overlapping elements (O(E^2))
    E = rm_overlapping_elems(V, E)

    return V, E

#======================================================================#
#
