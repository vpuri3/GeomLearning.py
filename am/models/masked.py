#
import torch
from torch import nn
import torch_geometric as pyg

__all__ = [
    'MaskedLoss',
    'MaskedModel',
]

#======================================================================#
class MaskedLoss(torch.nn.Module):
    """
    Loss function that:
    1.  masks out predictions made at the final step because
        next step prediction doesn't make any sense there.
    2.  Computes batch loss over the active elements of the graph,
        i.e., regions where batch.mask == True.
    """
    def __init__(self, mask: bool):
        super().__init__()

        self.tol = 1e-4
        self.mask = mask
        self.lossfun = nn.MSELoss()

    def forward(self, model, batch):
        yh = model(batch)

        # (1)
        last_step_mask = (batch.t <= 1. - self.tol).view(-1, 1)
        yh = yh * last_step_mask

        # (2)
        if self.mask:
            mask = batch.mask.view(-1,1)
            denom = mask.sum() * batch.y.size(1)
            loss = (mask * (yh - batch.y) ** 2).sum() / denom
        else:
            loss = self.lossfun(yh, batch.y)

        return loss

#======================================================================#
class MaskedModel(nn.Module):
    """
    Model that masks out predictions made at the final step
    because next step prediction doesn't make any sense there.
    """
    def __init__(self, model, mask=True, mask_bulk=False):
        super().__init__()
        self.mask = mask
        self.model = model
        self.mask_bulk = mask_bulk

    @torch.no_grad()
    def reduce_graph(self, graph):

        if hasattr(graph, 'edge_index'):
            if graph.edge_index is None:
                return graph
        else:
            return graph

        edge_index, edge_attr = pyg.utils.subgraph(
            graph.mask.view(-1), graph.edge_index, graph.edge_attr
        )

        return pyg.data.Data(
            **{k: graph[k] for k in graph.keys() if k != 'edge_index' and k != 'edge_attr'},
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

    def forward(self, graph):

        if self.mask:
            mask = graph.mask.view(-1, 1)
            subgraph = self.reduce_graph(graph)
            x = self.model(subgraph)
            x = x * mask
        else:
            x = self.model(graph)

        if self.mask_bulk:
            mask_bulk = graph.mask_bulk.view(-1, 1)
            x = x * mask_bulk

        return x

#======================================================================#
#