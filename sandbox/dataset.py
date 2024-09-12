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
    split=[.8, .2],
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

    point = torch.stack(point, dim=point[0].ndim) # channel dim
    value = torch.stack(value, dim=value[0].ndim)

    if datatype == "pointcloud":
        point = point.flatten(0, 2)
        value = value.flatten(0, 2)

        inside = torch.argwhere(M3D.reshape(-1)).reshape(-1)
        point = point[inside]
        value = value[inside]

    elif datatype == "image":
        # batches over time-dimension
        pass
    elif datatype == "mesh":
        # get inside points only and create adjacency matrix
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    data = TensorDataset(point, value)
    return torch.utils.data.random_split(data, split)

# class SandboxDataset(Dataset):
#     def __init__(self, nx, nz, nt, nw1=None, nw2=None):
#         self.shape = SandboxShape(nx, nz, nt, nw1=nw1, nw2=nw2)
#         return
#
#     def __len__(self):
#         return (self.shape.nx * self.shape.nz * self.shape.nt)
#
#     def __getitems__(self, idx):
#         idx = torch.tensor(idx).to(torch.int) # list -> tensor
#
#         nx = self.shape.nx
#         nz = self.shape.nz
#         nt = self.shape.nt
#
#         # get indices
#         it  = torch.floor(idx / (nx * nz)) % nt
#         ixz = idx - (it * nt)
#         iz  = torch.floor(ixz / nx)        % nz # extra % never hurts
#         ix  = (ixz % nz)                   % nx
#
#         x = self.shape.x[ix.to(int)]
#         z = self.shape.z[iz.to(int)]
#         t = self.shape.t[it.to(int)]
#
#         mask, temp, disp, sdf = self.shape.fields(x, z, t)
#
#         point = torch.stack([x, z, t], dim=1)
#         value = torch.stack([temp], dim=1)
#
#         # return point, value
#         return [(point[i], value[i]) for i in range(point.size(0))]

#
