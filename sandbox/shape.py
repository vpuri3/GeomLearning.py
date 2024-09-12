#
import torch

import matplotlib.pyplot as plt
import matplotlib.animation as anim

__all__ = [
    'Shape',
]

class Shape:
    """
     Hourglass SDF

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

