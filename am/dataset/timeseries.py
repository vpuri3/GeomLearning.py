import torch
import torch_geometric as pyg
import torch.multiprocessing as mp

import scipy
from tqdm import tqdm

import os
import copy
import json
from typing import Union

from .utils import (timeseries_dataset, merge_timeseries)
from .transform import DatasetTransform

__all__ = [
    'TimeseriesDatasetTransform',
    'TimeseriesDataset',
    'split_timeseries_dataset',
]

#======================================================================#
# TRANSFORM
#======================================================================#
class TimeseriesDatasetTransform(DatasetTransform):
    def __init__(
        self,
        disp=True, vmstr=True, temp=True,
        mesh=True, elems=False, orig=False,
        metadata=False, merge=True, interpolate=True,
    ):

        super().__init__(
            disp=disp, vmstr=vmstr, temp=temp,
            mesh=mesh, elems=elems, orig=orig, metadata=metadata,
        )

        self.merge = merge
        self.interpolate = interpolate

        return

    def makefields(self, data):
        '''
        used in am.time_march
        '''

        xs = []
        xs = [*xs, data.disp[:,2].reshape(-1,1)] if self.disp  else xs
        xs = [*xs, data.vmstr.reshape(-1,1)    ] if self.vmstr else xs
        xs = [*xs, data.temp.reshape(-1,1)     ] if self.temp  else xs

        return torch.cat(xs, dim=-1) / self.scale.to(xs[0].device)

    @torch.no_grad()
    def interpolate_layer(self, u: torch.tensor, graph, istep: int, tol=1e-4):
        '''
        fill `u` in between `istep` and `istep+1`
        '''

        if not self.merge:
            return u

        # ensure not final step
        assert istep + 1 != graph.metadata['time_steps']

        z = graph.pos[:,2].reshape(-1)
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

        first_step = istep == 0
        last_step  = (istep + 1) == nsteps

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
        dz = 1
        fmin = 0.1
        zi = md['zmax'][istep]
        zz = (graph.pos[:, 2] - zi + 20 * dz) / (self.pos_scale[2] / 10)
        mask_bulk = fmin + (1 + torch.tanh(zz)) * (1 - fmin) / 2

        # fmin = 0.1
        # zi = md['zmax'][istep]
        # zz = (graph.pos[:,2] - zi + 10) / (self.pos_scale[2] * 10)
        # mask_bulk = fmin + (1 + torch.tanh(zz)) * (1 - fmin) / 2

        # normalize fields
        pos, disp, vmstr, temp, edge_dxyz = self.normalize_fields(graph)

        # time
        t  = torch.full((N, 1), graph.metadata['t_val'])
        dt = torch.full((N, 1), graph.metadata['dt_val'])

        # fields (works for merged=True)
        if self.merge:
            if not last_step:

                disp0  = disp[ istep, :, 2].reshape(-1,1)
                vmstr0 = vmstr[istep, :, 0].reshape(-1,1)
                temp0  = temp[ istep, :, 0].reshape(-1,1)

                disp1  = disp[ istep+1, :, 2].reshape(-1,1)
                vmstr1 = vmstr[istep+1, :, 0].reshape(-1,1)
                temp1  = temp[ istep+1, :, 0].reshape(-1,1)

                if self.interpolate:
                    disp0  = self.interpolate_layer(disp0,  graph, istep, tol=tol)
                    vmstr0 = self.interpolate_layer(vmstr0, graph, istep, tol=tol)
                    temp0  = self.interpolate_layer(temp0,  graph, istep, tol=tol)

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
        else:
            # look back
            if not first_step:
                pass

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

        assert len(ys) == self.nfields, f"At least one of disp, vmstr, temp must be True. Got {self.disp}, {self.vmstr}, {self.temp}."

        x = torch.cat(xs, dim=-1)
        y = torch.cat(ys, dim=-1)

        edge_attr = edge_dxyz
        data = self.make_pyg_data(graph, edge_attr, x=x, y=y, t=t, mask=mask, mask_bulk=mask_bulk,)

        return data

#======================================================================#
# TIMESERIES DATASET
#======================================================================#
class TimeseriesDataset(pyg.data.Dataset):
    def __init__(
        self, root, transform=None, force_reload=False,
        merge=None, num_workers=None, exclude_list=None,
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
        if exclude_list is not None:
            self.case_files = [c for c in self.case_files if c not in exclude_list]

        with open(os.path.join(root, 'series.json')) as file:
            time_step_dict = json.load(file)
        self.time_steps = torch.tensor(
            [time_step_dict[case_file[:-3]] for case_file in self.case_files])
        self.time_steps_cum = self.time_steps.cumsum(0)

        super().__init__(root, transform, force_reload=force_reload)

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

        mp.set_start_method('spawn', force=True)
        with mp.Pool(self.num_workers) as pool:
            list(tqdm(pool.imap_unordered(self.process_single, icases), total=num_cases))

        return

    def process_single(self, icase):
        case_file = self.case_files[icase]
        dataset = timeseries_dataset(self.raw_paths[icase])
        if self.merge:
            graph = merge_timeseries(dataset, case_file[:-3])
            torch.save(graph, self.processed_paths[icase])
            del graph
        else:
            torch.save(dataset, self.processed_paths[icase])

        del dataset
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
        graph = torch.load(path, weights_only=False, mmap_mode='r')

        if not self.merge:
            graph = graph[time_step]

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

#======================================================================#
# SPLIT TIMESERIES DATASET
#======================================================================#
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
#
