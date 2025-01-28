#
import torch
import torch_geometric as pyg

import scipy
import numpy as np

__all__ = [
    # dataset utilities
    'makegraph',
    'timeseries_dataset',
    'merge_timeseries',
    'get_zmax_list',

    # mesh interpolation
    'interpolate_idw',
    'combine_meshes',
    'make_finest_mesh',
]

#======================================================================#
# DATASET UTILITIES
#======================================================================#

def makegraph(data, case_name=None, time_steps=None):
    '''
    Arguments:
    - data: dict of np.arrays containing the relevant fields
    - case_name: str case identifier
    - time_steps: number of time-steps (overwritten for merged datasets)
    '''

    # fields
    verts = torch.tensor(data['verts'], dtype=torch.float)   # [Nv, 3]
    elems = torch.tensor(data['elems'], dtype=torch.int)     # [Ne, 8]
    temp  = torch.tensor(data['temp'] , dtype = torch.float) # [Nv, 1]
    disp  = torch.tensor(data['disp'] , dtype = torch.float) # [Nv, 3]
    vmstr = torch.tensor(data['von_mises_stress'], dtype = torch.float) # [Nv, 1]

    if 'zmax' in data:
        zmax = torch.tensor(data['zmax'])
    else:
        zmax = None

    # edges
    elems = elems - torch.min(elems)    # ensures zero indexing
    connectivity = [                    # hexa8 elements
        (0, 1), (1, 2), (2, 3), (3, 0), # cube base
        (4, 5), (5, 6), (6, 7), (7, 4), # cube top
        (0, 4), (1, 5), (2, 6), (3, 7), # vertical edges
    ]

    edges = set()
    for elem in elems:
        for (i, j) in connectivity:
            edge1 = (elem[i].item(), elem[j].item())
            edge2 = (elem[j].item(), elem[i].item())

            edges.add(edge1)
            edges.add(edge2)

    edge_index = torch.tensor(list(edges))           # [Nedges, 2]
    edge_index = edge_index.T.contiguous()           # [2, Nedges]
    edge_index = pyg.utils.coalesce(edge_index)      # remove duplicate edges
    edge_index = pyg.utils.to_undirected(edge_index) # guarantee bidirectionality

    # edge features
    dx = verts[edge_index[0], 0] - verts[edge_index[1], 0]
    dy = verts[edge_index[0], 1] - verts[edge_index[1], 1]
    dz = verts[edge_index[0], 2] - verts[edge_index[1], 2]

    edge_dxyz = torch.stack([dx, dy, dz], dim=-1) # [Nedge, 3]

    if disp.ndim == 3: # merged timeseries
        time_steps = disp.shape[0]

    metadata = {
        "case_name"  : case_name,  # str
        'zmax'       : zmax,       # list
        "time_steps" : time_steps, # int
    }

    # make graph
    graph = pyg.data.Data(
        metadata=metadata,
        edge_index=edge_index, elems=elems,           # connectivity
        temp=temp, disp=disp, vmstr=vmstr, pos=verts, # nodal fields
        edge_dxyz=edge_dxyz,                          # edge  fields
    )

    return graph

#======================================================================#
def timeseries_dataset(case_file: str):
    assert case_file.endswith('.pt'), f"got invalid file name {case_file}"
    case_name = os.path.basename(case_file)[:-3]

    case = torch.load(case_file, weights_only=False)
    nsteps = len(case['verts'])

    dataset = []
    for i in range(nsteps):
        step = dict(verts=case['verts'][i], elems=case['elems'][i],
                    temp =case['temp' ][i], disp =case['disp' ][i],
                    von_mises_stress=case['von_mises_stress'][i])
        graph = makegraph(step, case_name, nsteps)
        dataset.append(graph)

    return dataset

#======================================================================#
def merge_timeseries(dataset, case_name=None, tol=1e-6):
    # output graph
    V, E  = make_finest_mesh(dataset)
    V, E  = V.numpy(force=True), E.numpy(force=True)
    N, NV = len(dataset), V.shape[0]

    # layer heights
    zmax = np.array(get_zmax_list(dataset))

    temps = []
    disps = []
    vmstrs = []

    for i in range(N):
        _pos   = dataset[i].pos.numpy(force=True)
        _temp  = dataset[i].temp.numpy(force=True)
        _disp  = dataset[i].disp.numpy(force=True)
        _vmstr = dataset[i].vmstr.numpy(force=True)

        temp  = np.zeros((NV, 1))
        disp  = np.zeros((NV, 3))
        vmstr = np.zeros((NV, 1))

        mask = (V[:,2] <= zmax[i] + tol).reshape(-1)
        temp[ mask] = interpolate_idw(_pos, _temp , V[mask])
        disp[ mask] = interpolate_idw(_pos, _disp , V[mask])
        vmstr[mask] = interpolate_idw(_pos, _vmstr, V[mask])

        temps.append(temp)
        disps.append(disp)
        vmstrs.append(vmstr)

    temp  = np.stack(temps , axis=0)
    disp  = np.stack(disps , axis=0)
    vmstr = np.stack(vmstrs, axis=0)

    data = dict(verts=V, elems=E, temp=temp, disp=disp, von_mises_stress=vmstr,
                zmax=zmax)
    graph = makegraph(data, case_name)

    return graph

#======================================================================#
def get_zmax_list(dataset):
    zmax = []
    for graph in dataset:
        zm = graph.pos[:,2].max().item()
        zmax.append(zm)
    return zmax

#======================================================================#
# MESH INTERPOLATION
#======================================================================#
def interpolate_idw(x_src, u_src, x_dst, k=4, pow=2, tol=1e-6, workers=-1, tree=None):
    # IDW = Inverse Distance Weighting
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
