#
import torch
from torch import nn

import torch_scatter
import torch_geometric as pyg
from torch_geometric.nn import MessagePassing

__all__ = [
    'MaskedMGN',
    #
    'MeshGraphNet',
    "MeshGraphLayer",
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
# MeshGraphNets
# https://medium.com/stanford-cs224w/learning-mesh-based-flow-simulations-on-graph-networks-44983679cf2d
#======================================================================#
class MeshGraphNet(nn.Module):
    def __init__(self, ci_node, ci_edge, co, hidden_dim, num_layers, **kwargs):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(ci_node, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(ci_edge, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.processor = nn.ModuleList()
        for _ in range(num_layers):
            layer = MeshGraphLayer(hidden_dim, hidden_dim, **kwargs)
            self.processor.append(layer)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, co),
        )

        return

    def forward(self, data):
        xn, xe = data.x, data.edge_attr
        edge_index = data.edge_index

        xn = self.node_encoder(xn)
        xe = self.edge_encoder(xe)

        for layer in self.processor:
            xn, xe = layer(xn, xe, edge_index)

        xn = self.decoder(xn)

        return xn
#

class MeshGraphLayer(MessagePassing):
    def __init__(self, ci, co, **kwargs):
        super().__init__(**kwargs)

        self.node_mlp = nn.Sequential(
            nn.Linear(2 * ci, co),
            nn.ReLU(),
            nn.Linear(co, co),
            nn.LayerNorm(co),
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * ci, co),
            nn.ReLU(),
            nn.Linear(co, co),
            nn.LayerNorm(co),
        )

        return

    def forward(self, xn, xe, edge_index):

        msg, ye = self.propagate(edge_index, x=xn, edge_attr=xe, dim_size=xn.size(0))

        yn = torch.cat([msg, xn], dim=1) # cat messages with node features
        yn = self.node_mlp(yn)           # apply node MLP
        yn = yn + xn                     # residual connection

        return yn, ye

    def message(self, x_i, x_j, edge_attr):

        xe = edge_attr
        ye = torch.cat([x_i, x_j, xe], dim=1) # cat node and edge features
        ye = self.edge_mlp(ye)                # apply edge MLP
        ye = xe + ye                          # residual connection

        return ye

    def aggregate(self, ye, edge_index, dim_size=None):
        node_dim = 0
        msg = torch_scatter.scatter(
            ye, edge_index[0,:], dim=node_dim,
            dim_size=dim_size, reduce='sum',
        )

        return msg, ye
#======================================================================#
#
