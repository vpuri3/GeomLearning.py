import torch
from torch import nn

import mlutils

__all__ = [
    'MaskedMGN',
]

#======================================================================#
class MaskedMGN(nn.Module):
    def __init__(self, ci, ce, co, w, num_layers, apply_mask=True):
        super().__init__()
        self.apply_mask = apply_mask
        self.gnn = mlutils.MeshGraphNet(ci, ce, co, w, num_layers)

    def reduce_graph(self, data):
        return data
        # # remove edges corresponding to nodes imask==False
        # mask  = data.mask
        # imask = torch.where(mask > 1e-4)
        # x = data.x
        # edge_index = data.edge_index
        # edge_attr = data.edge_attr

        return pyg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def forward(self, data, tol=1e-6):
        graph = self.reduce_graph(data)
        x = self.gnn(graph)

        if self.apply_mask:
            mask = data.mask.view(-1, 1)
            x = x * mask

        last_step_mask = (data.t <= 1. - tol).view(-1, 1)
        x = x * last_step_mask

        return x

#======================================================================#
