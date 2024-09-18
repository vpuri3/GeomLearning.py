#
import torch
from torch.utils.data import Dataset, TensorDataset

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
    elif datatype == "point-image":
        # batches over time-dimension
        point = torch.stack(point, dim=1) # [N, C, W, H]
        point = point[:, :, 0, 0]         # [N, C]

        value = torch.stack(value, dim=1) # [N, C, W, H]
        value = value * M3D.unsqueeze(1)
    elif datatype == "image":
        # batches over time-dimension
        point = torch.stack(point, dim=1) # [N, C, H, W]
        value = torch.stack(value, dim=1)
        value = value * M3D.unsqueeze(1)
    elif datatype == "image-nextstep":
        # batches over time-dimension
        point = torch.stack(point, dim=1)
        value = torch.stack(value, dim=1) # [N, C, H, W]
        # point = point * M3D.unsqueeze(1)
        value = value * M3D.unsqueeze(1)

        point = point[0:-1] 
        value = value[1:] - value[0:-1]
    elif datatype == "graph":

        assert mask == "finaltime"

        point = torch.stack(point, dim=point[0].ndim) # [Nt, Nz, Nx, C]
        value = torch.stack(value, dim=value[0].ndim)

        edges, idx_lin, idx_cart = shape.create_final_graph()
        iz, ix =  idx_cart[:, 0], idx_cart[:, 1]

        # node features # (Temp, time)
        point = point[:, ix, iz, :] # [Nt, V, C]
        value = value[:, ix, iz, :]

        # edge features # (x-relative, z-relative)
        x = x.reshape(self.nt, -1)
        z = z.reshape(self.nt, -1)

        edge_x = x[:, edges[:, 0]] - x[:, edges[1]]
        edge_z = z[:, edges[:, 0]] - z[:, edges[1]]

        edge_features = torch.stack([edge_x, edge_z], dim=1)

        raise NotImplementedError(f"Mesh datatype not implemented.")
    else:
        raise NotImplementedError(f"Got incompatible datatype: {datatype}.")

    data = TensorDataset(point, value)

    if split is None:
        return data
    else:
        return torch.utils.data.random_split(data, split)
#
