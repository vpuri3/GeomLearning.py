#
import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

__all__ = [
    'Shape',
]

class Shape:
    """
     Hourglass shape

       ---------------------
       |                   |
       |    ___________    |
       |    \         /    |
       |     \       /     |
       |      \     /      |
       |      |     |      |
       |      |     |      |
       |      /     \      |
       |     /       \     |  ^ z
       |    /         \    |  |
       ---------------------  L--> x

     `x ∈ [-1, 1]`
     `z ∈ [ 0, 1]`
     `t ∈ [ 0, 1]`

    """
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

    def fields(self, x, z, t):
        radius = self.nw1 * (self.nw2 - torch.sin(torch.pi * z))
        global_mask = torch.lt(torch.abs(x), radius)
        time_mask = torch.lt(z, t)

        mask = global_mask * time_mask
        temp = 1 + z - torch.sin(t)
        disp = torch.zeros(x.size()) * torch.nan

        bdist = torch.abs(z)
        tdist = torch.abs(z - t)
        rdist = torch.abs(torch.abs(x) - radius)

        tdist += 1e10 * (~global_mask)
        rdist += 1e10 * (~time_mask)

        dist = torch.minimum(bdist, tdist)
        dist = torch.minimum(dist , rdist)
        sign = 1 - 2 * mask
        sdf  = dist * sign

        return mask, temp * mask, disp * mask, sdf

    def fields_dense(self):
        t, z, x = torch.meshgrid([self.t, self.z, self.x], indexing='ij')
        fields = self.fields(x, z, t)
        return (x, z, t), fields

    def plot(self, nt_plt = 5):
        _, (mask, temp, disp, sdf) = self.fields_dense()

        # equivalent to `x.detatch().cpu().numpy()`
        x = self.x.numpy(force=True)
        z = self.z.numpy(force=True)
        t = self.t.numpy(force=True)

        mask = mask.numpy(force=True)
        temp = temp.numpy(force=True)
        disp = disp.numpy(force=True)
        sdf  = sdf.numpy(force=True)

        it_plt = torch.linspace(0, self.nt-1, nt_plt)
        it_plt = torch.round(it_plt).to(torch.int).numpy(force=True)

        fig, axs = plt.subplots(ncols=nt_plt, nrows=3, figsize = (15, 9))

        axs[0, 0].set_ylabel(f"Mask")
        axs[1, 0].set_ylabel(f"Temperature")
        axs[2, 0].set_ylabel(f"SDF")

        # # for debugging SDF
        # sdf = np.abs(sdf) < 1e-2

        for (i, it) in enumerate(it_plt):
            axs[0, i].set_title(f"Time {t[it].item():>5f}")

            p0 = axs[0, i].contourf(x, z, mask[it, :, :], levels= 1, cmap='Grays')
            p1 = axs[1, i].contourf(x, z, temp[it, :, :], levels=20, cmap='viridis')
            p2 = axs[2, i].contourf(x, z,  sdf[it, :, :], levels=20, cmap='viridis')

            fig.colorbar(p0, ax=axs[0, i])
            fig.colorbar(p1, ax=axs[1, i])
            fig.colorbar(p2, ax=axs[2, i])
        #

        #========================#
        fig.tight_layout()
        plt.savefig("fig1.png")

        # ani = anim.FuncAnimation(
        #     fig, update, frames=range(self.nt), interval=50
        # )

        return fig
#

