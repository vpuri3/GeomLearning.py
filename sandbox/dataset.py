#
import torch
from torch.utils.data import Dataset

import math

class SandboxDataset(Dataset):
    def __init__(self, Nx, Nz): # (Nw1, Nw2)

        # spilt = "train" / "test"
        # reserve shapes for test split

        self.Nx = Nx
        self.Nz = Nz
        # self.Nw1 = Nw1
        # self.Nw2 = Nw2

        self.x = torch.linspace(-1, 1, Nx)
        self.z = torch.linspace( 0, 1, Nz)

        return

    def __len__(self):
        return (self.Nx * self.Nz)
        # return (self.Nx * self.Nz * self.Nw1 * self.Nw2)

    def __getitem__(self, idx):
        # sdf, temp, displ

        ix = idx % self.Nz
        iz = math.floor(idx / self.Nz)

        x = torch.tensor([self.x[ix], self.z[iz]])
        u = torch.tensor([torch.sum(x), torch.diff(x)])

        return x, u
#

def makedata(Nx, Nz, Nw1=None, Nw2=None):
    # _data = SandboxDataset()
    # data_ = SandboxDataset()

    data = SandboxDataset(Nx, Nz)
    _data, data_ = torch.utils.data.random_split(data, [0.8, 0.2])

    return _data, data_
#
