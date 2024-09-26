#
import torch

import numpy as np

import networkx as nx
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
    def __init__(self, nx, nz, nt, nw1=None, nw2=None, blend=False):
        self.nx = nx
        self.nz = nz
        self.nt = nt

        if nw1 is None:
            nw1 = 0.5
        if nw2 is None:
            nw2 = 1.3

        self.nw1 = nw1
        self.nw2 = nw2

        self.blend = blend

        self.x = torch.linspace(-1, 1, nx)
        self.z = torch.linspace( 0, 1, nz)
        self.t = torch.linspace( 0, 1, nt)

        return

    def dt(self):
        return 1 / (self.nt - 1)

    def radius(self, z):
        return self.nw1 * (self.nw2 - torch.sin(torch.pi * z))

    def side_mask(self, x, z):
        return torch.lt(torch.abs(x), self.radius(z))

    def time_mask(self, z, t):
        if self.blend:
            d = t - z
            return torch.sigmoid(80 * d)
        else:
            return torch.lt(z, t)

    def mask(self, x, z, t):
        return self.side_mask(x, z) * self.time_mask(z, t)

    def sdf(self, x, z, t, tol=1e-6):
        radius = self.radius(z)
        side_mask_bool = self.side_mask(x, z) > tol
        time_mask_bool = self.time_mask(z, t) > tol

        mask_bool = side_mask_bool * time_mask_bool

        bdist = torch.abs(z)
        tdist = torch.abs(z - t)
        rdist = torch.abs(torch.abs(x) - radius)

        tdist += 1e10 * (~side_mask_bool)
        rdist += 1e10 * (~time_mask_bool)

        dist = torch.minimum(bdist, tdist)
        dist = torch.minimum(dist , rdist)
        sign = 1 - 2 * mask_bool

        return dist * sign

    def fields(self, x, z, t, tol=1e-6):
        mask = self.mask(x, z, t)
        temp = 1 + z - torch.sin(t)
        disp = torch.zeros(x.size()) * torch.nan
        sdf  = self.sdf(x, z, t, tol=tol)

        temp *= mask
        disp *= mask

        return mask, temp, disp, sdf

    def fields_dense(self):
        t, z, x = torch.meshgrid([self.t, self.z, self.x], indexing='ij')
        fields = self.fields(x, z, t)

        return (x, z, t), fields

    def final_mask(self):
        _, (M, _, _, _) = self.fields_dense()
        return M[-1]

    def linear_index(self, ix, iz):
        return iz * self.nx + ix

    def cartesian_index(self, idx):
        ix = idx  % self.nx
        iz = idx // self.nx
        return (ix, iz)

    def compute_final_graph(self, bidirectional=True, debug=False):
        M = self.final_mask()

        idx_cart = torch.argwhere(M)
        iz, ix =  idx_cart[:, 0], idx_cart[:, 1]

        # left/right/top/bottom neighbors
        _ix, ix_ = ix - 1, ix + 1
        _iz, iz_ = iz - 1, iz + 1

        # map invalid indices to typemax(int)
        if debug:
            print(f"Number of invalid indices")
            print(len(torch.argwhere(_ix < 0)))
            print(len(torch.argwhere(_iz < 0)))
            print(len(torch.argwhere(ix_ >= self.nx)))
            print(len(torch.argwhere(iz_ >= self.nz)))

        int_max = torch.iinfo(ix.dtype).max
        _ix[torch.argwhere(_ix < 0)] = int_max
        _iz[torch.argwhere(_iz < 0)] = int_max
        ix_[torch.argwhere(ix_ >= self.nx)] = int_max
        iz_[torch.argwhere(iz_ >= self.nz)] = int_max

        # compute linear indices
        idx_lin = self.linear_index( ix, iz )
        lft_lin = self.linear_index(_ix, iz )
        rgt_lin = self.linear_index(ix_, iz )
        top_lin = self.linear_index( ix, _iz)
        btm_lin = self.linear_index( ix, iz_)

        # only consider neighbors that are in idx_cart
        lft_in = torch.isin(lft_lin, idx_lin)
        rgt_in = torch.isin(rgt_lin, idx_lin)
        top_in = torch.isin(top_lin, idx_lin)
        btm_in = torch.isin(btm_lin, idx_lin)

        # get indices of valid neighbors
        ilft = torch.argwhere(lft_in)
        irgt = torch.argwhere(rgt_in)
        itop = torch.argwhere(top_in)
        ibtm = torch.argwhere(btm_in)

        # create edge if neighbor is in idx_cart
        lft_edges = torch.cat([idx_lin[ilft], lft_lin[ilft]], dim=1)
        rgt_edges = torch.cat([idx_lin[irgt], rgt_lin[irgt]], dim=1)
        top_edges = torch.cat([idx_lin[itop], top_lin[itop]], dim=1)
        btm_edges = torch.cat([idx_lin[ibtm], btm_lin[ibtm]], dim=1)

        if not bidirectional:
            # prune between left/right, up/down
            raise NotImplementedError()

        edges = torch.cat([
            lft_edges,
            rgt_edges,
            top_edges,
            btm_edges,
        ], dim=0)

        return edges, idx_lin, idx_cart

    @torch.no_grad()
    def plot_final_graph(self, debug=False):
        edges, _, idx_cart = self.compute_final_graph(debug=debug)

        edges    = edges.numpy(force=True)
        idx_cart = idx_cart.numpy(force=True)

        # background graph
        bg_graph = nx.Graph()
        bg_nodes = [(ix, iz) for ix in range(self.nx) for iz in range(self.nz)]
        bg_edgeH = [((ix, iz), (ix+1, iz)) for ix in range(self.nx-1) for iz in range(self.nz)]
        bg_edgeV = [((ix, iz), (ix, iz+1)) for ix in range(self.nx) for iz in range(self.nz-1)]
        
        bg_graph.add_nodes_from(bg_nodes)
        bg_graph.add_edges_from(bg_edgeH)
        bg_graph.add_edges_from(bg_edgeV)

        # active graph
        graph = nx.DiGraph()
        nodes = [(idx[1], idx[0]) for idx in idx_cart]
        _ix, _iz = self.cartesian_index(edges[:, 0])
        ix_, iz_ = self.cartesian_index(edges[:, 1])

        edges = [((_ix[i], _iz[i]), (ix_[i], iz_[i])) for i in range(len(_ix))]

        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        # make plot
        fig, ax = plt.subplots(1, 1)

        bg_pos = {(ix, iz): (self.x[ix], self.z[iz]) for (ix, iz) in bg_graph.nodes()}
        nx.draw(bg_graph, bg_pos, ax=ax,
            node_color="gray", edge_color="gray", node_size=10, # with_labels=True, font_size=8,
        )

        pos = {(ix, iz): (self.x[ix], self.z[iz]) for (ix, iz) in graph.nodes()}
        nx.draw(graph, pos, ax=ax,
            node_color="red", edge_color="black", node_size=10,
        )

        ax.axhline(y=0, color='black', linestyle='--')
        ax.axvline(x=0, color='black', linestyle='--')

        # ax.grid(True)
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")

        return fig

    @torch.no_grad()
    def plot(self, nt_plt = 5, animate = False):
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

        if animate:
            # ani = anim.FuncAnimation(
            #     fig, update, frames=range(self.nt), interval=50
            # )
            pass
        #


        return fig

    @torch.no_grad()
    def plot_compare(self, pred, nt_plt=5, nextstep=False):

        _, (_, temp, _, _) = self.fields_dense()

        assert pred.shape == temp.shape

        fig, axs = plt.subplots(ncols=5, nrows=3, figsize = (15, 9))
        axs[0, 0].set_ylabel(f"True")
        axs[1, 0].set_ylabel(f"Pred")
        axs[2, 0].set_ylabel(f"Errr")

        x = self.x.numpy(force=True)
        z = self.z.numpy(force=True)
        t = self.t.numpy(force=True)

        errr = torch.abs(pred - temp)
        pred = pred.numpy(force=True)
        errr = errr.numpy(force=True)

        it_plt = torch.linspace(0, self.nt-1, nt_plt)
        it_plt = torch.round(it_plt).to(torch.int).numpy(force=True)

        for (i, it) in enumerate(it_plt):
            axs[0, i].set_title(f"Time {t[it].item():2f}")

            p0 = axs[0, i].contourf(x, z, temp[it, :, :], levels=20, cmap='viridis')
            p1 = axs[1, i].contourf(x, z, pred[it, :, :], levels=20, cmap='viridis')
            p2 = axs[2, i].contourf(x, z, errr[it, :, :], levels=20, cmap='viridis')

            for j in range(2):
                axs[j, i].set_xlabel('')
                axs[j, i].set_xticks([])

            if i != 0:
                for j in range(3):
                    axs[j, i].set_ylabel('')
                    axs[j, i].set_yticks([])

            fig.colorbar(p0, ax=axs[0, i])
            fig.colorbar(p1, ax=axs[1, i])
            fig.colorbar(p2, ax=axs[2, i])

        return fig

    @torch.no_grad()
    def plot_history(self, nz=5):
        x = torch.zeros(nz, self.nt)
        z = torch.linspace(0, 1, nz).reshape(-1, 1) * torch.ones(1, self.nt)
        t = torch.ones(nz, 1) * self.t.reshape(1, -1)

        _, temp, _, _ = self.fields(x, z, t)

        fig, ax = plt.subplots(1,1)

        for iz in range(nz):
            ax.plot(self.t, temp[iz, :], label=f"z={z[iz, 0].item():.2e}")

        ax.set_xlabel("Time")
        ax.set_ylabel("Temperature")
        ax.legend(loc="center left")
        ax.set_title("Temperature history")

        return fig

    @torch.no_grad()
    def plot_distribution(self, nt=5):
        x = torch.zeros(nt, self.nz)
        z = torch.ones(nt, 1) * self.z.reshape(1, -1)
        t = torch.linspace(0, 1, nt).reshape(-1, 1) * torch.ones(1, self.nz)

        _, temp, _, _ = self.fields(x, z, t)

        fig, ax = plt.subplots(1,1)

        for it in range(nt):
            ax.plot(self.z, temp[it, :], label=f"t={t[it, 0].item():.2e}")

        ax.set_xlabel("Z-Position")
        ax.set_ylabel("Temperature")
        ax.legend(loc="center right")
        ax.set_title("Temperature distribution along centerline")

        return fig
#
