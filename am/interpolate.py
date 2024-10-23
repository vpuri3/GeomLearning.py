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
def interpolate_idw(x_src, u_src, x_dst, k=4, pow=2, tol=1e-6, workers=-1):
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

def rm_overlapping_elems(verts, elems):
    mins, maxs = bounding_box(verts, elems)

    # O(N^2) check
    contained = is_contained(
        mins.unsqueeze(1), maxs.unsqueeze(1), # i
        mins.unsqueeze(0), maxs.unsqueeze(0), # j
    )
    contained.diagonal().mul_(False)

    ij = torch.argwhere(contained)
    idx_rm = torch.unique(ij[:,1])

    idx_keep = [i for i in range(elems.shape[0]) if i not in idx_rm]
    elems_refined = elems[idx_keep]

    return elems_refined

#======================================================================#
def combine_meshes(verts1, elems1, verts2, elems2):
    verts = torch.cat([verts1, verts2], dim=0)
    verts, idx = torch.unique(verts, dim=0, return_inverse=True)

    e1 = idx[:len(verts1)][elems1]
    e2 = idx[len(verts1):][elems2]

    elems = torch.cat([e1, e2], dim=0)
    elems = elems.unique(dim=0)

    # rm redundant elements
    elems_sort = elems.sort(dim=1)[0]
    idx_unique = torch.unique(elems_sort, dim=0, return_inverse=True)[1]
    idx_unique = torch.unique(idx_unique, sorted=False)

    elems = elems[idx_unique]

    # rm overlapping elements
    elems = rm_overlapping_elems(verts, elems)

    return verts, elems

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
# def make_finest_mesh(dataset, outdir, icase, tol=1e-6, workers=-1):
#     N = len(dataset)
#     V = dataset[-1].pos.numpy(force=True)
#     E = dataset[-1].elems.numpy(force=True)
#
#     tree = scipy.spatial.KDTree(V)
#
#     for i in range(N-1):
#         verts = dataset[i].pos
#         elems = dataset[i].elems
#
#         # find novel vertices
#         dist, idx_near = tree.query(verts, k=1, workers=workers)
#         idx_new = np.argwhere(dist > tol).reshape(-1)
#         if len(idx_new) > 0:
#             continue
#         verts = verts[idx_new]
#
#         # remove overlapping elements form E (use idx_near)
#
#         # print(len(inew))
#         # print(vert_idx[inew].shape)
#         # print(verts[vert_idx[inew]].shape)
#         # V, E = combine_meshes(V, E, verts, elems)
#     #
#
#     return V, E

#======================================================================#
#
