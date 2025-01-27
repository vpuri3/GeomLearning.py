#
import torch
from torch import nn
import torch_geometric as pyg

from .meshGNN import MeshGraphNet

__all__ = [
    'MaskedMGN',
]

#======================================================================#
__all__.append('MaskedLoss')

class MaskedLoss(torch.nn.Module):
    def __init__(self, mask: bool):
        super().__init__()

        self.tol = 1e-4
        self.mask = mask
        self.lossfun = nn.MSELoss()

    def forward(self, model, batch):
        yh = model(batch)

        # zero out predictions made at the final step
        # because next step prediction doesn't make any sense there
        last_step_mask = (batch.t <= 1. - self.tol).view(-1, 1)
        yh = yh * last_step_mask

        # assume mask has been applied to target and prediction 
        if self.mask:
            mask = batch.mask.view(-1,1)
            denom = mask.sum() * batch.y.size(1)
            loss = (mask * (yh - batch.y) ** 2).sum() / denom
        else:
            loss = self.lossfun(yh, batch.y)

        return loss

#======================================================================#
class MaskedMGN(nn.Module):
    def __init__(self, ci, ce, co, w, num_layers, mask=True, mask_bulk=False):
        super().__init__()
        self.mask = mask
        self.mask_bulk = mask_bulk
        self.gnn = MeshGraphNet(ci, ce, co, w, num_layers)

    @torch.no_grad()
    def reduce_graph(self, graph):

        edge_index, edge_attr = pyg.utils.subgraph(
            graph.mask.view(-1), graph.edge_index, graph.edge_attr
        )

        return pyg.data.Data(
            x=graph.x, edge_index=edge_index, edge_attr=edge_attr
        )

    def forward(self, graph, tol=1e-6):

        if self.mask:
            mask = graph.mask.view(-1, 1)
            subgraph = self.reduce_graph(graph)
            x = self.gnn(subgraph)
            x = x * mask
        else:
            x = self.gnn(graph)

        if self.mask_bulk:
            mask_bulk = graph.mask_bulk.view(-1, 1)
            x = x * mask_bulk

        return x

#======================================================================#
#
