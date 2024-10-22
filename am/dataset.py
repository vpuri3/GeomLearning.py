import torch
import torch_geometric as pyg
import numpy as np
from tqdm import tqdm

import os

import mlutils

__all__ = [
    'GraphDataset',
    'makegraph',
    'timeseries_dataset',
]

class GraphDataset(pyg.data.Dataset):
    def __init__(
        self, root, transform=None, pre_transform=None,
        pre_filter=None, force_reload=False
    ):
        self.cases = [c for c in os.listdir(root) if c.endswith('.npz')]
        super().__init__(root, transform, pre_transform,
                         pre_filter, force_reload=force_reload)

    @property
    def raw_paths(self):
        return [os.path.join(self.root, self.cases[i]) for i in range(len(self))]

    @property
    def processed_paths(self):
        proc_dir = os.path.join(self.root, "processed")
        cases = [f"case{str(i).zfill(5)}_{self.cases[i][:-4]}.pt" for i in range(len(self))]
        return [os.path.join(proc_dir, case) for case in cases]

    def process(self):
        raw_paths = self.raw_paths
        proc_paths = self.processed_paths
        for idx in tqdm(range(len(self))):
            data = np.load(raw_paths[idx])
            graph = makegraph(data)
            torch.save(graph, proc_paths[idx])
        return

    def len(self):
        return len(self.cases)

    def get(self, idx):
        path = self.processed_paths[idx]
        return torch.load(path, weights_only=False)
#

def makegraph(data):

    elems = torch.tensor(data['elems'], dtype=torch.int)     # [Ne, 8]
    verts = torch.tensor(data['verts'], dtype=torch.float)   # [Nv, 3]
    temp  = torch.tensor(data['temp'] , dtype = torch.float) # [Nv, 1]
    disp  = torch.tensor(data['disp'] , dtype = torch.float) # [Nv, 3]
    vmstr = torch.tensor(data['von_mises_stress'], dtype = torch.float) # [Nv, 1]

    # get edges
    elems = elems - 1 # fix indexing

    connectivity = [                    # hexa8 elements
        (0, 1), (1, 2), (2, 3), (3, 4), # cube base
        (4, 5), (5, 6), (6, 7), (7, 4), # cube top
        (0, 4), (1, 5), (2, 6), (3, 7), # vertical edges
    ]

    edges = set()
    for voxel in elems:
        for (i, j) in connectivity:
            edge1 = (voxel[i].item(), voxel[j].item())
            edge2 = (voxel[j].item(), voxel[i].item())

            edges.add(edge1)
            edges.add(edge2)

    edge_index = torch.tensor(list(edges))           # [Nedges, 2]
    edge_index = edge_index.T.contiguous()           # [2, Nedges]
    edge_index = pyg.utils.coalesce(edge_index)      # remove duplicate edges
    edge_index = pyg.utils.to_undirected(edge_index) # guarantee bidirectionality

    # node attributes
    x = torch.cat([verts], dim=-1)
    # y = torch.cat([vmstr], dim=-1)
    y = torch.cat([temp ], dim=-1)

    # edge attributes
    edge_dx = verts[edge_index[0], 0] - verts[edge_index[1], 0]
    edge_dy = verts[edge_index[0], 1] - verts[edge_index[1], 1]
    edge_dz = verts[edge_index[0], 2] - verts[edge_index[1], 2]

    edge_attr = torch.stack([edge_dx, edge_dy, edge_dz], dim=-1) # [Nedge, 3]

    # normalize
    xbar, xstd = mlutils.mean_std(x, -1)
    ybar, ystd = mlutils.mean_std(y, -1)
    ebar, estd = mlutils.mean_std(edge_attr, -1)

    metadata = {
        "xbar" : xbar, "xstd" : xstd,
        "ybar" : ybar, "ystd" : ystd,
        "ebar" : ebar, "estd" : estd,
    }

    # x = mlutils.normalize(x, xbar, xstd)
    # y = mlutils.normalize(y, ybar, ystd)
    # edge_attr = mlutils.normalize(edge_attr, ebar, estd)


    data = pyg.data.Data(
        x=x, y=y, edge_index=edge_index, edge_attr=edge_attr,
        pos=verts, elems=elems, **metadata,
    )

    return data
#

def timeseries_dataset(case_file: str):
    assert case_file.endswith('.pt'), f"got invalid file name {case_file}"

    case = torch.load(case_file, weights_only=False)
    nsteps = len(case['verts'])

    dataset = []
    for i in range(nsteps):
        step = dict(verts=case['verts'][i], elems=case['elems'][i],
                    temp =case['temp' ][i], disp =case['disp' ][i],
                    von_mises_stress=case['von_mises_stress'][i])
        graph = makegraph(step)
        dataset.append(graph)

    return dataset
#
