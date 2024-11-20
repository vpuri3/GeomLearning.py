import torch
import torch_geometric as pyg
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm

import os
import json
from typing import Union

import mlutils
from .utils import get_zmax_list
from .interpolate import interpolate_idw, make_finest_mesh

__all__ = [
    'FinaltimeDataset',
    'TimeseriesDataset',
    'makegraph',
    'timeseries_dataset',
    'merge_timeseries',
]

#======================================================================#
class TimeseriesDataset(pyg.data.Dataset):
    def __init__(
        self, root, transform=None, pre_transform=None,
        pre_filter=None, force_reload=False,
        merge=None, num_workers = 8,
    ):
        """
        Create dataset of time-series

        Arguments:
        - `merge`: return fields on graph made by merging all the timeseries
        meshes.
        """
        self.merge = merge
        self.num_workers = num_workers
        self.case_files = [c for c in os.listdir(root) if c.endswith('.pt')]
        # self.filter = None

        with open(os.path.join(root, 'series.json')) as file:
            time_step_dict = json.load(file)
        self.time_steps = torch.tensor(
            [time_step_dict[case_file[:-3]] for case_file in self.case_files])
        self.time_steps_cum = self.time_steps.cumsum(0)

        super().__init__(root, transform, pre_transform,
                         pre_filter, force_reload=force_reload)

    @property
    def raw_paths(self):
        return [os.path.join(self.root, case_file) for case_file in self.case_files]

    @property
    def processed_paths(self):
        proc_dir  = self.proc_dir()
        os.makedirs(proc_dir, exist_ok=True)
        case_files = [f"case{str(i).zfill(5)}_{case_file}"
            for (i, case_file) in enumerate(self.case_files)]
        return [os.path.join(proc_dir, case) for case in case_files]

    def proc_dir(self):
        proc_name = 'processed_merged' if self.merge else 'processed'
        return os.path.join(self.root, proc_name)

    def process(self):
        num_cases = len(self.case_files)
        icases = range(num_cases)

        with mp.Pool(self.num_workers) as pool:
            list(tqdm(pool.imap_unordered(self.process_single, icases), total=num_cases))

        # for icase in tqdm(range(num_cases)):
        #     self.process_single(icase)

        return

    def process_single(self, icase):
        case_file = self.case_files[icase]
        dataset = timeseries_dataset(self.raw_paths[icase])
        if self.merge:
            graph = merge_timeseries(dataset, case_file[:-3])
            torch.save(graph, self.processed_paths[icase])
        else:
            torch.save(dataset, self.processed_paths[icase])
        return

    def len(self):
        return sum(self.time_steps)

    def case_range(self, case: Union[int, str]):
        if isinstance(case, int):
            i0 = 0 if case == 0 else self.time_steps_cum[case-1]
            i1 = self.time_steps_cum[case]
            assert i1-i0 == self.time_steps[case]
            return range(i0, i1)
        else: # str
            icase = self.case_files.index(case + '.pt')
            return self.case_range(icase)

    def get(self, idx):
        # get case and time step
        icase = torch.argwhere(idx < self.time_steps_cum)[0].item()
        nprev = 0 if icase == 0 else self.time_steps_cum[icase-1].item()
        time_step = idx - nprev

        # GET PATH
        path = self.processed_paths[icase]

        # LOAD GRAPH
        if self.merge:
            graph = torch.load(path, weights_only=False)
        else:
            graph = torch.load(path, weights_only=False)[time_step]

        # WRITE ACTIVE TIME-STEP (zero indexed)
        graph.metadata['time_step'] = time_step

        return graph
#

#======================================================================#
class FinaltimeDataset(pyg.data.Dataset):
    def __init__(
        self, root, transform=None, pre_transform=None,
        pre_filter=None, force_reload=False
    ):
        self.case_files = [c for c in os.listdir(root) if c.endswith('.npz')]
        super().__init__(root, transform, pre_transform,
                         pre_filter, force_reload=force_reload)

    @property
    def raw_paths(self):
        return [os.path.join(self.root, case_file)
            for case_file in self.case_files]

    @property
    def processed_paths(self):
        proc_dir = os.path.join(self.root, "processed")
        case_files = [f"case{str(i).zfill(5)}_{self.case_files[i][:-4]}.pt" for i in range(len(self))]
        # case_files = [f"case{str(i).zfill(5)}_{case[:-4]}.pt" for (i, case) in enumerate(self.case_files)]
        return [os.path.join(proc_dir, case) for case in case_files]

    def process(self):
        raw_paths = self.raw_paths
        proc_paths = self.processed_paths
        for i in tqdm(range(len(self))):
            data = np.load(raw_paths[i])
            graph = makegraph(data, self.case_files[i][:-4], 1)
            torch.save(graph, proc_paths[i])
        return

    def len(self):
        return len(self.case_files)

    def get(self, idx):
        path = self.processed_paths[idx]
        return torch.load(path, weights_only=False)

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
        # case identifier
        "case_name"  : case_name,
        # time stepping
        'zmax' : zmax,
        "time_steps" : time_steps,
        # mean, std deviation
        'pos'     : (verts_bar, verts_std),
        'disp'    : (disp_bar , disp_std ),
        'temp'    : (temp_bar , temp_std ),
        'vmstr'   : (vmstr_bar, vmstr_std),
        # extrema
        'extrema' : (verts_min, verts_max),
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

#======================================================================#
#
