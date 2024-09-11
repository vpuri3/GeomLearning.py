#
import torch
from torch.utils.data import Dataset, TensorDataset

import matplotlib.pyplot as plt
import matplotlib.animation as anim

import math

#
# Hourglass SDF
# 
#   ---------------------
#   |                   |
#   |    ___________    |
#   |    \         /    |
#   |     \       /     |
#   |      \     /      |
#   |      |     |      |
#   |      |     |      |
#   |      /     \      |
#   |     /       \     |  ^ z
#   |    /         \    |  |
#   ---------------------   --> x
# 
# `x ∈ [-1, 1]`
# `z ∈ [ 0, 1]`
# `t ∈ [ 0, 1]`
#

class SandboxShape:
    def __init__(self, nx, nz, nt, nw1=None, nw2=None):
        self.nx = nx
        self.nz = nz
        self.nt = nt

        if nw1 is None:
            nw1 = 0.5
        if nw2 is None:
            nw2 = 1.3

        self.nw1 = nw1
        self.nw2 = nw2

        self.x = torch.linspace(-1, 1, nx)
        self.z = torch.linspace( 0, 1, nz)
        self.t = torch.linspace( 0, 1, nt)

        return

    # @property
    # def nx(self):
    #     return self.nx
    #
    # @property
    # def nz(self):
    #     return self.nz
    #
    # @property
    # def nt(self):
    #     return self.nt

    def fields(self, x, z, t):
        """
        assumes axisymmetric SDF

        Julia code
        ```
        function fields(x, z, t)
            r = @. 0.5f0 * (1.3f0 - sin(pi32 * z)) # radius
            s = abs.(x) .≤ r                       # full SDF
            M = @. z ≤ t                           # time mask
            s = s .* M
        
            d = rand(size(s)...)
            T = @. 1f0 + z - sin(t)
        
            s, d .* s, T .* s
        end
        ```
        """
        radius = self.nw1 * (self.nw2 - torch.sin(torch.pi * z))
        global_sdf = torch.lt(torch.abs(x), radius) # abs(x) ≤ radius
        time_mask = torch.lt(z, t) # z ≤ t

        sdf  = global_sdf * time_mask
        temp = 1 + z - torch.sin(t)
        disp = torch.zeros(x.size()) * torch.nan

        return sdf, temp * sdf, disp * sdf

    def fields_dense(self):
        t, z, x = torch.meshgrid([self.t, self.z, self.x], indexing='ij')
        sdf, temp, disp = self.fields(x, z, t)
        return (x, z, t), (sdf, temp, disp)

    def plot(self, nt_plt = 5):
        _, (sdf, temp, disp) = self.fields_dense()

        # equivalent to `x.detatch().cpu().numpy()`
        x = self.x.numpy(force=True)
        z = self.z.numpy(force=True)
        t = self.t.numpy(force=True)
        sdf  = sdf.numpy(force=True)
        temp = temp.numpy(force=True)
        disp = disp.numpy(force=True)

        it_plt = torch.linspace(0, self.nt-1, nt_plt)
        it_plt = torch.round(it_plt).to(torch.int).numpy(force=True)

        fig, axs = plt.subplots(ncols=nt_plt, nrows=2, figsize = (12, 6))

        axs[0, 0].set_ylabel(f"SDF")
        axs[1, 0].set_ylabel(f"Temperature")

        for (i, it) in enumerate(it_plt):
            axs[0, i].set_title(f"Time {t[it].item():>5f}")

            axs[0, i].contourf(x, z, sdf[it, :, :], cmap='greys')
            axs[1, i].contourf(x, z, temp[it, :, :], cmap='viridis')

            # axs[0, i].colorbar()
            # axs[1, i].colorbar()
        #

        #========================#
        fig.tight_layout()
        plt.savefig("fig1.png")

        # ani = anim.FuncAnimation(
        #     fig, update, frames=range(self.nt), interval=50
        # )

        return fig
#

class SandboxDataset(Dataset):
    def __init__(self, nx, nz, nt, nw1=None, nw2=None):
        self.shape = SandboxShape(nx, nz, nt, nw1=nw1, nw2=nw2)
        return

    def __len__(self):
        return (self.shape.nx * self.shape.nz * self.shape.nt)

    def __getitems__(self, idx):
        idx = torch.tensor(idx).to(torch.int) # list -> tensor

        nx = self.shape.nx
        nz = self.shape.nz
        nt = self.shape.nt

        # get indices
        it  = torch.floor(idx / (nx * nz)) % nt
        ixz = idx - (it * nt)
        iz  = torch.floor(ixz / nx)        % nz # extra % never hurts
        ix  = (ixz % nz)                   % nx

        x = self.shape.x[ix.to(int)]
        z = self.shape.z[iz.to(int)]
        t = self.shape.t[it.to(int)]

        sdf, temp, disp = self.shape.fields(x, z, t)

        point = torch.stack([x, z, t], dim=1)
        value = torch.stack([temp], dim=1)

        # return point, value
        return [(point[i], value[i]) for i in range(point.size(0))]
#

def makedata(
    nx, nz, nt, nw1=None, nw2=None,
    insideonly=True,
):
    #---------------------------#
    # data = SandboxDataset(nx, nz, nt, nw1=nw1, nw2=nw2)
    #---------------------------#
    shape = SandboxShape(nx, nz, nt, nw1=nw1, nw2=nw2)
    xzt, fields = shape.fields_dense() # [nt, nz, nx]

    x, z, t = [a.reshape(-1) for a in xzt]
    sdf, temp, disp = [a.reshape(-1) for a in fields]

    point = torch.stack([x, z, t], dim=1)
    value = torch.stack([temp], dim=1)

    if insideonly:
        inside = torch.argwhere(sdf)
        point = point[inside]
        value = value[inside]

    data = TensorDataset(point, value)
    #---------------------------#

    _data, data_ = torch.utils.data.random_split(data, [0.8, 0.2])
    return _data, data_
#
