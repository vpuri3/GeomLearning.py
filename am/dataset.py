import torch
import torch_geometric as pyg
import torch.multiprocessing as mp

import scipy
import numpy as np
from tqdm import tqdm

import os
import copy
import json
from typing import Union

import mlutils
from .utils import get_zmax_list
from .interpolate import interpolate_idw, make_finest_mesh

__all__ = [

    # timeseries
    'MergedTimeseriesDataTransform',
    'TimeseriesDataset',
    'split_timeseries_dataset',

    # final time
    'FinaltimeDataset',

    # utilities
    'makegraph',
    'timeseries_dataset',
    'merge_timeseries',
]

#======================================================================#
# TIMESERIES DATASET
#======================================================================#
class MergedTimeseriesDataTransform:
    def __init__(
        self,
        disp=True, vmstr=True, temp=True, # fields
        interpolate=True,
    ):

        # fields
        self.disp  = disp
        self.vmstr = vmstr
        self.temp  = temp

        self.interpolate = interpolate

        # pos  : x, y [-30, 30] mm, z [-25, 60] mm ([-25, 0] build plate)
        # disp : x [-0.5, 0.5] mm, y [-0.05, 0.05] mm, z [-0.1, -1] mm
        # vmstr: [0, ~5e3] Pascal (?)
        # temp : Celcius [25, ~300]
        # time: [0, 1]

        self.pos_scale = torch.tensor([30., 30., 60.])
        self.disp_scale  = 1.
        self.vmstr_scale = 1000.
        self.temp_scale  = 500.

        # makefields
        self.nfields = disp + vmstr + temp 

        scale = []
        scale = [*scale, self.disp_scale ] if self.disp  else scale
        scale = [*scale, self.vmstr_scale] if self.vmstr else scale
        scale = [*scale, self.temp_scale ] if self.temp  else scale
        self.scale = torch.tensor(scale)

        assert self.nfields == len(scale)

        return

    def makefields(self, data):
        '''
        used in am.time_march
        '''

        xs = []
        xs = [*xs, data.disp[:,2].view(-1,1)] if self.disp  else xs
        xs = [*xs, data.vmstr.view(-1,1)    ] if self.vmstr else xs
        xs = [*xs, data.temp.view(-1,1)     ] if self.temp  else xs

        return torch.cat(xs, dim=-1) / self.scale.to(xs[0].device)

    @torch.no_grad()
    def interpolate_up(self, u: torch.tensor, graph, istep: int, tol=1e-4):
        '''
        fill `u` in between `istep` and `istep+1`
        '''

        # ensure not final step
        assert istep + 1 != graph.metadata['time_steps']

        z = graph.pos[:,2].view(-1)
        zmax = graph.metadata['zmax']
        z0 = zmax[istep-1] if istep != 0 else 0
        z1 = zmax[istep]
        z2 = zmax[istep+1]

        idx0 = (z0 - tol <= z) * (z <= z1 + tol) # [istep-1, istep]
        idx1 = (z1 + tol <  z) * (z <= z2 + tol) # (istep, istep+1]

        out = scipy.interpolate.griddata(
            graph.pos[idx0].numpy(force=True),
            u[idx0].numpy(force=True),
            graph.pos[idx1].numpy(force=True),
            method='nearest',
        )

        u.clone()
        u[idx1, :] = torch.tensor(out, dtype=torch.float, device=u.device)

        return u

    def __call__(self, graph, tol=1e-4):

        N  = graph.pos.size(0)
        md = graph.metadata
        istep  = md['time_step']
        nsteps = md['time_steps']
        last_step = (istep + 1) == nsteps

        #
        # TODO:
        #    dz = zmax[istep+1] - zmax[istep]
        #
        # use dz to decide the interface width such that
        # interface fully encompasses one layer and ends at the next.
        # input to GNN should not have sharp discontinuity

        #
        # OBSERVATION:
        #
        # large errors concentrated in regions with bottleneck
        # like a contracting / expanding nozzle
        #
        # what to do about it?
        #

        # interface mask
        if not last_step:
            zmax = md['zmax'][istep+1]
            mask = graph.pos[:,2] <= (zmax + tol)
        else:
            zmax = md['zmax'][-1]
            mask = torch.full((N,), True)

        # bulk mask
        # dz = 1
        # fmin = 0.1
        # zi = md['zmax'][istep]
        # zz = (graph.pos[:, 2] - zi + 20 * dz) / (self.pos_scale[2] / 10)
        # mask_bulk = fmin + (1 + torch.tanh(zz)) * (1 - fmin) / 2

        fmin = 0.1
        zi = md['zmax'][istep]
        zz = (graph.pos[:,2] - zi + 10) / (self.pos_scale[2] * 10)
        mask_bulk = fmin + (1 + torch.tanh(zz)) * (1 - fmin) / 2

        # normalize fields
        pos   = graph.pos   / self.pos_scale
        disp  = graph.disp  / self.disp_scale
        vmstr = graph.vmstr / self.vmstr_scale
        temp  = graph.temp  / self.temp_scale
        edge_dxyz = graph.edge_dxyz / self.pos_scale

        # time
        t  = torch.full((N, 1), graph.metadata['t_val'])
        dt = torch.full((N, 1), graph.metadata['dt_val'])

        # fields
        if not last_step:

            disp0  = disp[ istep, :, 2].view(-1,1)
            vmstr0 = vmstr[istep, :, 0].view(-1,1)
            temp0  = temp[ istep, :, 0].view(-1,1)

            disp1  = disp[ istep+1, :, 2].view(-1,1)
            vmstr1 = vmstr[istep+1, :, 0].view(-1,1)
            temp1  = temp[ istep+1, :, 0].view(-1,1)

            if self.interpolate:
                disp0  = self.interpolate_up(disp0,  graph, istep, tol=tol)
                vmstr0 = self.interpolate_up(vmstr0, graph, istep, tol=tol)
                temp0  = self.interpolate_up(temp0,  graph, istep, tol=tol)

            disp_in  = disp0
            vmstr_in = vmstr0
            temp_in  = temp0

            disp_out  = disp1  - disp0
            vmstr_out = vmstr1 - vmstr0
            temp_out  = temp1  - temp0

        else:
            disp_in  = torch.zeros((N, 1))
            disp_out = torch.zeros((N, 1))

            vmstr_in  = torch.zeros((N, 1))
            vmstr_out = torch.zeros((N, 1))

            temp_in  = torch.zeros((N, 1))
            temp_out = torch.zeros((N, 1))

        # features / labels
        xs = [pos, t, dt,]
        ys = []

        if self.disp:
            xs.append(disp_in)
            ys.append(disp_out)
        if self.vmstr:
            xs.append(vmstr_in)
            ys.append(vmstr_out)
        if self.temp:
            xs.append(temp_in)
            ys.append(temp_out)

        assert len(ys) > 0, f"At least one of disp, vmstr, temp must be True. Got {self.disp}, {self.vmstr}, {self.temp}."

        x = torch.cat(xs, dim=-1)
        y = torch.cat(ys, dim=-1)

        edge_attr = edge_dxyz

        return pyg.data.Data(
            x=x, y=y, t=t, mask=mask, mask_bulk=mask_bulk,
            edge_attr=edge_attr, edge_index=graph.edge_index, elems=graph.elems,
            disp=graph.disp[istep], vmstr=graph.vmstr[istep], temp=graph.temp[istep], pos=graph.pos,
            metadata=graph.metadata,
        )

#======================================================================#
class TimeseriesDataset(pyg.data.Dataset):
    def __init__(
        self, root, transform=None, pre_transform=None,
        pre_filter=None, force_reload=False,
        merge=None, num_workers=None,
    ):
        """
        Create dataset of time-series

        Arguments:
        - `merge`: return fields on graph made by merging all the timeseries
        meshes.
        """
        if num_workers is None:
            self.num_workers = mp.cpu_count() // 2
        else:
            self.num_workers = num_workers

        self.merge = merge
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
        return [os.path.join(proc_dir, case_file) for case_file in self.case_files]

    def proc_dir(self):
        proc_name = 'processed_merged' if self.merge else 'processed'
        return os.path.join(self.root, proc_name)

    def process(self):
        num_cases = len(self.case_files)
        icases = range(num_cases)

        # for icase in tqdm(range(num_cases)):
        #     self.process_single(icase)

        with mp.Pool(self.num_workers) as pool:
            list(tqdm(pool.imap_unordered(self.process_single, icases), total=num_cases))

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
            if not case.endswith('.pt'):
                case = case + '.pt'
            icase = self.case_files.index(case)
            return self.case_range(icase)

    def get(self, idx):
        # get case and time step
        icase = torch.argwhere(idx < self.time_steps_cum)[0].item()
        nprev = 0 if icase == 0 else self.time_steps_cum[icase-1].item()
        time_step  = idx - nprev
        time_steps = self.time_steps[icase]

        # GET PATH
        path = self.processed_paths[icase]

        # LOAD GRAPH
        if self.merge:
            graph = torch.load(path, weights_only=False)
        else:
            graph = torch.load(path, weights_only=False)[time_step]

        # WRITE ACTIVE TIME-STEP (zero indexed)
        assert time_steps == graph.metadata['time_steps'] > 0

        if time_steps == 1:
            dt = 0.
            t  = 1.
        else:
            dt = 1 / (time_steps - 1)
            t  = time_step * dt

        graph.metadata['t_val']     = t
        graph.metadata['dt_val']    = dt
        graph.metadata['time_step'] = time_step

        return graph

def split_timeseries_dataset(dataset, split=None, indices=None):
    num_cases = len(dataset.case_files)

    if indices is None:
        indices = torch.utils.data.random_split(range(num_cases), split)

    num_split = len(indices)

    # deepcopy
    subsets = [copy.deepcopy(dataset) for _ in range(num_split)]

    for s in range(num_split):
        idxs   = list(indices[s])
        subset = subsets[s]

        subset.case_files = [subset.case_files[idx] for idx in idxs]
        subset.time_steps = torch.tensor([subset.time_steps[idx].item() for idx in idxs])
        subset.time_steps_cum = subset.time_steps.cumsum(0)

    return subsets

#======================================================================#
# FINALTIME DATASET
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
