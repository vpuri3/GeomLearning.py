import torch
import torch_geometric as pyg
import torch.multiprocessing as mp

import scipy
import numpy as np
from tqdm import tqdm

import os

__all__ = [
    'FinaltimeDatasetTransform',
    'FinaltimeDataset',
]

#======================================================================#
# TRANSFORM
#======================================================================#
class FinaltimeDatasetTransform:
    def __init__(
        self,
        # fields
        disp=True, vmstr=True, temp=True, mesh=True,
        metadata=False,
    ):

        # fields
        self.disp  = disp
        self.vmstr = vmstr
        self.temp  = temp
        self.mesh  = mesh

        self.metadata = metadata

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

    def __call__(self, graph, tol=1e-4):

        N  = graph.pos.size(0)
        md = graph.metadata

        # normalize fields
        pos   = graph.pos   / self.pos_scale
        disp  = graph.disp  / self.disp_scale
        vmstr = graph.vmstr / self.vmstr_scale
        temp  = graph.temp  / self.temp_scale
        edge_dxyz = graph.edge_dxyz / self.pos_scale

        # only consider z disp
        disp = disp[:, 2]

        # features / labels
        xs = [pos,]
        ys = []

        if self.disp:
            ys.append(disp)
        if self.vmstr:
            ys.append(vmstr)
        if self.temp:
            ys.append(temp)

        assert len(ys) > 0, f"At least one of disp, vmstr, temp must be True. Got {self.disp}, {self.vmstr}, {self.temp}."

        x = torch.cat(xs, dim=-1)
        y = torch.cat(ys, dim=-1)

        edge_attr = edge_dxyz

        data = pyg.data.Data(
            x=x, y=y, t=t,
            # edge_attr=edge_attr, edge_index=graph.edge_index, elems=graph.elems,
            disp=graph.disp, vmstr=graph.vmstr, temp=graph.temp, pos=graph.pos,
        )

        if self.mesh:
            data.elems      = graph.elems
            data.edge_attr  = edge_attr
            data.edge_index = graph.edge_index

        if self.metadata:
            data.metadata = graph.metadata

        return data

#======================================================================#
# FINALTIME DATASET
#======================================================================#
class FinaltimeDataset(pyg.data.Dataset):
    def __init__(
        self, root, transform=None, pre_transform=None,
        pre_filter=None, force_reload=False,
        num_workers=None,
    ):

        if num_workers is None:
            self.num_workers = mp.cpu_count() // 2
        else:
            self.num_workers = num_workers

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
        num_cases = len(self.case_files)
        icases = range(num_cases)

        with mp.Pool(self.num_workers) as pool:
            list(tqdm(pool.imap_unordered(self.process_single, icases), total=num_cases))

        # for icase in tqdm(range(len(self))):
        #     self.process_single(icase)

        return

    def process_single(self, icase):
        data = np.load(self.raw_paths[icase])
        graph = makegraph(data, self.case_files[icase][:-4], 1)
        torch.save(graph, self.processed_paths[icase])
        return

    def len(self):
        return len(self.case_files)

    def get(self, idx):
        path = self.processed_paths[idx]
        return torch.load(path, weights_only=False)

#======================================================================#
#
