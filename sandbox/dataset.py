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
        point = torch.stack(point, dim=point[0].ndim) # channel dim
        value = torch.stack(value, dim=value[0].ndim) # at the end

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
    elif datatype == "mesh":
        # get inside points only and create adjacency matrix
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    data = TensorDataset(point, value)

    if split is None:
        return data
    else:
        return torch.utils.data.random_split(data, split)
#
