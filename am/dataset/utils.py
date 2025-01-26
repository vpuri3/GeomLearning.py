#
import torch
import numpy as np

from .mesh_interpolate import interpolate_idw, make_finest_mesh

__all__ = [
    'makegraph',
    'timeseries_dataset',
    'merge_timeseries',
    'get_zmax_list',
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

    # normalization
    verts_bar, verts_std = mlutils.mean_std(verts, -1)
    disp_bar , disp_std  = mlutils.mean_std(disp , -1)
    temp_bar , temp_std  = mlutils.mean_std(temp , -1)
    vmstr_bar, vmstr_std = mlutils.mean_std(vmstr, -1)
    # extrema
    verts_min, verts_max = verts.aminmax(dim=0, keepdim=True)

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

def get_zmax_list(dataset):
    zmax = []
    for graph in dataset:
        zm = graph.pos[:,2].max().item()
        zmax.append(zm)
    return zmax

#======================================================================#
