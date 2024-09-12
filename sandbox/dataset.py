#
import torch
from torch.utils.data import Dataset, TensorDataset

__all__ = [
    'makedata',
]

def add_fields_to_list(string, fields):
    assert len(string) != 0
    x, z, t, S, T, D = fields

    ll = []
    ll.append(x) if 'x' in string else None
    ll.append(z) if 'z' in string else None
    ll.append(t) if 't' in string else None

    ll.append(S) if 'S' in string else None
    ll.append(T) if 'T' in string else None
    ll.append(D) if 'D' in string else None

    return ll

def PointwiseDataset(
        shape, insideonly=False, inputs="xzt", outputs="ST",
):
    xzt, fields = shape.fields_dense() # [nt, nz, nx]

    x, z, t = [a.reshape(-1) for a in xzt]
    S, T, D = [a.reshape(-1) for a in fields] # SDF, Temp, Disp

    fields = [x, z, t, S, T, D]
    point = add_fields_to_list(inputs , fields) # "xztSTD"
    value = add_fields_to_list(outputs, fields)

    point = torch.stack(point, dim=1)
    value = torch.stack(value, dim=1)

    if insideonly:
        inside = torch.argwhere(S).reshape(-1)
        point = point[inside]
        value = value[inside]

    return TensorDataset(point, value)

def makedata(shape, **kwargs):
    data = PointwiseDataset(shape, **kwargs)
    return torch.utils.data.random_split(data, [0.8, 0.2])

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
#         sdf, temp, disp = self.shape.fields(x, z, t)
#
#         point = torch.stack([x, z, t], dim=1)
#         value = torch.stack([temp], dim=1)
#
#         # return point, value
#         return [(point[i], value[i]) for i in range(point.size(0))]

#
