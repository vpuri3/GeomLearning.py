#
import torch
import torch_geometric as pyg
from torch.utils.data import Dataset, TensorDataset

import mlutils

__all__ = [
    'makedata',
]

def add_fields_to_list(string, fields):
    assert len(string) != 0
    x, z, t, M, T, D, S = fields

    ll = []
    ll.append(x) if 'x' in string else None
    ll.append(z) if 'z' in string else None
    ll.append(t) if 't' in string else None

    ll.append(M) if 'M' in string else None
    ll.append(T) if 'T' in string else None
    ll.append(D) if 'D' in string else None
    ll.append(S) if 'S' in string else None

    return ll

def makedata(
    shape, sdf_clamp=1e-2,
    inputs="xzt", outputs="T",
    datatype="pointcloud",
    mask=None,
    split=None,
):
    # sample data
    (x, z, t), (M, T, D, S) = shape.fields_dense() # [nt, nz, nx]

    # clamp SDF
    S = torch.clamp(S, -sdf_clamp, sdf_clamp)

    # get relevant mask
    if mask == "spacetime":
        M3D = M
    elif mask == "finaltime":
        M2D = M[-1]
        M3D = M2D.unsqueeze(0) * torch.ones(M.size(0), 1, 1)
    elif mask == None:
        M3D = torch.fill(M, True)
    else:
        raise NotImplementedError()

    fields = [x, z, t, M3D, T, D, S]
    point = add_fields_to_list(inputs , fields) # "xztMTDS"
    value = add_fields_to_list(outputs, fields)

    if datatype == "pointcloud":
        point = torch.stack(point, dim=point[0].ndim) # [Nt, Nz, Nx, C]
        value = torch.stack(value, dim=value[0].ndim)

        point = point.flatten(0, 2) # [N, C]
        value = value.flatten(0, 2)

        inside = torch.argwhere(M3D.reshape(-1)).reshape(-1)
        point = point[inside]
        value = value[inside]
        channel_dim = 3
    elif datatype == "point-image":
        # batches over time-dimension
        point = torch.stack(point, dim=1) # [N, C, W, H]
        point = point[:, :, 0, 0]         # [N, C]

        value = torch.stack(value, dim=1) # [N, C, W, H]
        value = value * M3D.unsqueeze(1)
        channel_dim = 1
    elif datatype == "image":
        # batches over time-dimension
        point = torch.stack(point, dim=1) # [N, C, H, W]
        value = torch.stack(value, dim=1)
        value = value * M3D.unsqueeze(1)
        channel_dim = 1
    elif datatype == "image-nextstep":
        # batches over time-dimension
        point = torch.stack(point, dim=1)
        value = torch.stack(value, dim=1) # [N, C, H, W]
        # point = point * M3D.unsqueeze(1)
        value = value * M3D.unsqueeze(1)

        point = point[0:-1] 
        value = (value[1:] - value[0:-1]) / shape.dt()
        channel_dim = 1
    elif "graph" in datatype:
        assert mask == "finaltime"

        # stack features [Nt, Nxz, C]
        point = torch.stack(point, dim=-1)
        value = torch.stack(value, dim=-1)

        point = point.reshape(shape.nt, shape.nz * shape.nx, -1)
        value = value.reshape(shape.nt, shape.nz * shape.nx, -1)

        # create graph
        shape.create_final_graph()

        # node features and targets
        point = point[:, shape.glo_node_index, :] # [Nt, Nodes, C]
        value = value[:, shape.glo_node_index, :]

        # edge features # (x-relative, z-relative)
        x = x.reshape(shape.nt, -1)
        z = z.reshape(shape.nt, -1)

        edge_dx = x[0, shape.glo_edge_index[0]] - x[0, shape.glo_edge_index[1]]
        edge_dz = z[0, shape.glo_edge_index[0]] - z[0, shape.glo_edge_index[1]]
        edge_attr = torch.stack([edge_dx, edge_dz], dim=-1) # [Nedges, C]

        if "nextstep" in datatype:
            point = point[0:-1]
            value = (value[1:] - value[0:-1]) / shape.dt()

        channel_dim = 2
    else:
        raise NotImplementedError(f"Got incompatible datatype: {datatype}.")

    xbar, xstd = mlutils.mean_std(point, channel_dim)
    ybar, ystd = mlutils.mean_std(value, channel_dim)

    metadata = {
        "shape" : shape,
        # data
        "channel_dim" : channel_dim,
        "xbar" : xbar, "xstd" : xstd,
        "ybar" : ybar, "ystd" : ystd,
    }

    # point = mlutils.normalize(point, xbar, xstd)
    # value = mlutils.normalize(value, ybar, ystd)

    if "graph" in datatype:
        data = [
            pyg.data.Data(x=point[i], y=value[i],
                edge_index=shape.loc_edge_index, edge_attr=edge_attr,
            )
            for i in range(len(value))
        ]
        metadata = {**metadata, }
    else:
        data = TensorDataset(point, value)

    if split is None:
        return data, metadata
    else:
        return torch.utils.data.random_split(data, split), metadata
#
