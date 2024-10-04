#
# 3rd party
import torch
from torch import nn
import torch_geometric as pyg

import numpy as np
import matplotlib.pyplot as plt

# builtin
import random
import argparse

# local
import sandbox
import mlutils

def train_loop(model, data, E=100, lrs=None, nepochs=None, **kw):
    if lrs is None:
        lrs = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6]
    if nepochs is None:
        nepochs = [.1*E, .2*E, .25*E, .25*E, .1*E, .1*E]
        nepochs = [int(e) for e in nepochs]
    assert len(lrs) == len(nepochs)

    for i in range(len(lrs)):
        kw = {
            **kw,
            "lr" : lrs[i],
            "nepochs" : nepochs[i],
        }
        trainer = mlutils.Trainer(model, data, **kw)
        trainer.train()

    return model

class MaskedGNN(nn.Module):
    def __init__(self, shape, ci, co, w, num_layers):
        super().__init__()
        self.shape = shape
        self.act = nn.ReLU()
        self.encoder = pyg.nn.GCNConv(ci,  w)
        self.decoder = pyg.nn.GCNConv( w, co)
        self.processor = nn.ModuleList()
        for _ in range(num_layers):
            layer = pyg.nn.GCNConv(w, w)
            self.processor.append(layer)

    @torch.no_grad()
    def compute_mask(self, x):
        x0, z0, t0 = [x[:,c] for c in range(3)]
        t1 = t0 + self.shape.dt()
        M = self.shape.mask(x0, z0, t1)
        return M.unsqueeze(1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        M = self.compute_mask(x)
        
        x = self.encoder(x, edge_index)
        x = self.act(x)
        for layer in self.processor:
            x = layer(x, edge_index)
            x = self.act(x)
        x = self.decoder(x, edge_index)

        return x * M
#

class MaskedMGN(nn.Module):
    def __init__(self, shape, ci, co, w, num_layers):
        super().__init__()
        self.shape = shape
        self.gnn = mlutils.MeshGraphNet(ci, 2, co, w, num_layers)

    @torch.no_grad()
    def compute_mask(self, x):
        x0, z0, t0 = [x[:,c] for c in range(3)]
        t1 = t0 + self.shape.dt()
        M = self.shape.mask(x0, z0, t1)
        return M.unsqueeze(1)

    def forward(self, data):
        M = self.compute_mask(data.x)
        x = self.gnn(data)
        return x * M
#

def train_gnn_nextstep(device, outdir, resdir, name, blend=True, train=True):
    outname = outdir + "gnn_nextstep_" + name
    resname = resdir + "gnn_nextstep_" + name

    if blend:
        outname = outname + "_blend"
        resname = resname + "_blend"

    modelfile  = outname + ".pth"
    imagefile1 = resname + ".png"
    imagefile2 = resname + "_autoregressive" + ".png"

    # DATA
    if name == "hourglass":
        nw1 = None
        nw2 = None
    elif name == "alldomain":
        nw1 = torch.inf
        nw2 = torch.inf

    nx, nz, nt = 256, 256, 200
    shape = sandbox.Shape(nx, nz, nt, nw1, nw2, blend=blend)
    data, metadata = sandbox.makedata(
        shape, inputs="xztT", outputs="T", datatype="graph-nextstep", mask="finaltime",
    )

    # MODEL
    ci, co, w, num_layers = 4, 1, 128, 4
    model = MaskedMGN(shape, ci, co, w, num_layers)

    # TRAIN
    if train:
        train_loop(model, data, device=device, E=200, gnn=True, _batch_size=2,)
        torch.save(model.to("cpu").state_dict(), modelfile)

    # VISUALIZE
    model.eval()
    model.load_state_dict(torch.load(modelfile, weights_only=True))

    with torch.no_grad():
        dt = shape.dt()
        _nt = len(data) # nt-1
        num_nodes = data[0].num_nodes

        ####
        # NEXT STEP
        ####

        temp_prev = torch.stack([d.x[:,-1] for d in data], dim=0)
        dTdt = mlutils.eval_gnn(data, model, device, batch_size=4)
        dTdt = dTdt.reshape(_nt, num_nodes)
        
        temp = temp_prev + dTdt * dt
        temp = torch.cat([temp_prev[0].unsqueeze(0), temp], dim=0)
        
        pred1 = torch.zeros(nt, nz * nx)
        pred1[:, shape.glo_node_index] = temp
        pred1 = pred1.reshape(nt, nz, nx)
        
        fig1 = shape.plot_compare(pred1)
        fig1.savefig(imagefile1, dpi=300)

        ####
        # NEXT STEP AUTOREGRESSIVE
        ####

        K = 1 # nt // 8
        temps = []
        graph = data[0].clone()

        for k in range(K):
            temps.append(data[k].x[:, -1])
        for k in range(K-1, _nt):
            print(k)
            time = dt * k
            temp_prev = temps[-1]
            graph.x[:, -2] = time
            graph.x[:, -1] = temp_prev
            dTdt = model(graph).squeeze(-1)
            temp = temp_prev + dt * dTdt
            temps.append(temp)
        temp = torch.stack(temps, dim=0)

        pred2 = torch.zeros(nt, nz * nx)
        pred2[:, shape.glo_node_index] = temp
        pred2 = pred2.reshape(nt, nz, nx)

        fig2 = shape.plot_compare(pred2)
        fig2.savefig(imagefile2, dpi=300)

    return

def train_gnn(device, outdir, resdir, name, blend=True, train=True):
    outname = outdir + "gnn_" + name
    resname = resdir + "gnn_" + name

    if blend:
        outname = outname + "_blend"
        resname = resname + "_blend"

    modelfile  = outname + ".pth"
    imagefile1 = resname + ".png"
    imagefile2 = resname + "_autoregressive" + ".png"

    # DATA
    if name == "hourglass":
        nw1 = None
        nw2 = None
    elif name == "alldomain":
        nw1 = torch.inf
        nw2 = torch.inf

    nx, nz, nt = 256, 256, 200
    shape = sandbox.Shape(nx, nz, nt, nw1, nw2, blend=blend)
    data, metadata = sandbox.makedata(
        shape, inputs="xzt", outputs="T", datatype="graph", mask="finaltime",
    )

    # MODEL
    # ci, co, w, num_layers = 3, 1, 256, 4
    # model = MaskedGNN(shape, ci, co, w, num_layers)
    ci, co, w, num_layers = 3, 1, 128, 4
    model = MaskedMGN(shape, ci, co, w, num_layers)

    # TRAIN
    if train:
        train_loop(model, data, device=device, E=100, gnn=True, _batch_size=2,)
        torch.save(model.to("cpu").state_dict(), modelfile)

    # VISUALIZE
    model.eval()
    model.load_state_dict(torch.load(modelfile, weights_only=True))

    with torch.no_grad():
        num_nodes = data[0].num_nodes
        temp = mlutils.eval_gnn(data, model, device, batch_size=4)
        temp = temp.reshape(shape.nt, num_nodes)

        pred1 = torch.zeros(nt, nz * nx)
        pred1[:, shape.glo_node_index] = temp # 1
        pred1 = pred1.reshape(nt, nz, nx)

        fig1 = shape.plot_compare(pred1)
        fig1.savefig(imagefile1, dpi=300)

    return

class MaskedUNet(nn.Module):
    def __init__(self, shape, ci, co, k):
        super().__init__()
        self.shape = shape
        self.unet = mlutils.UNet(ci, co, k)

    @torch.no_grad()
    def compute_mask(self, x):
        x0, z0, t0 = [x[0,c,:,:] for c in range(3)]
        t1 = t0 + self.shape.dt()
        M = self.shape.mask(x0, z0, t1)
        return M.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        M = self.compute_mask(x)
        x = self.unet(x)
        return x * M
#

def train_cnn(device, outdir, resdir, name, blend=True, train=True):

    outname = outdir + "cnn_nextstep_" + name
    resname = resdir + "cnn_nextstep_" + name

    if blend:
        outname = outname + "_blend"
        resname = resname + "_blend"

    modelfile  = outname + ".pth"
    imagefile1 = resname + ".png"
    imagefile2 = resname + "_autoregressive" + ".png"

    # DATA
    if name == "hourglass":
        nw1 = None
        nw2 = None
    elif name == "alldomain":
        nw1 = torch.inf
        nw2 = torch.inf

    nx, nz, nt = 256, 256, 200
    shape = sandbox.Shape(nx, nz, nt, nw1, nw2, blend=blend)
    data, mean_std = sandbox.makedata(
        shape, inputs="xztT", outputs="T", datatype="image-nextstep", mask="finaltime",
    )

    # MODEL
    ci, co, k = 4, 1, 3
    model = MaskedUNet(shape, ci, co, k)

    # TRAIN
    if train:
        train_loop(model, data, device=device, _batch_size=4, E=100)
        torch.save(model.to("cpu").state_dict(), modelfile)

    # VISUALIZE
    model.eval()
    model.load_state_dict(torch.load(modelfile, weights_only=True))

    with torch.no_grad():
        xztT, _ = data[:]
        dt = shape.dt()

        ####
        # NEXT STEP
        ####

        pred1 = mlutils.eval_model(xztT, model, device, batch_size=4)
        pred1 = (dt * pred1 + xztT[:, -1].unsqueeze(1)).squeeze(1)
        pred1 = torch.cat([xztT[0, -1].unsqueeze(0), pred1], dim=0)

        fig1 = shape.plot_compare(pred1)
        fig1.savefig(imagefile1, dpi=300)

        ####
        # NEXT STEP AUTOREGRESSIVE
        ####

        # def process(y0, y1):
        #     xzt = y0[:, 0:3, :, :]
        #     temp = y0[:, -1, :, :].unsqueeze(1) + y1
        #     return torch.cat([xzt, temp], dim=1)
        #
        # def save(ys):
        #     temps = [y[:, -1, :, :] for y in ys]
        #     return torch.cat(temps, dim=0)
        # pred2 = mlutils.autoregressive_rollout(
        #     xztT[0].unsqueeze(0), model, len(data),
        #     process=process, save=save, device=device,
        # )

        K = nt // 8
        preds2 = []
        input = xztT[0].clone().unsqueeze(0) # [1, c, nz, nt]
        _nt = len(xztT)                    # shape.nt - 1

        for k in range(K):
            preds2.append(xztT[k, -1]) # [nz, nx]
        for k in range(K-1, _nt):
            print(k)
            time = dt * k
            temp_prev = preds2[-1]
            input[:, -2] = time
            input[:, -1] = temp_prev
            dTdt = model(input).squeeze(0).squeeze(0)
            pred2 = temp_prev + dt * dTdt
            preds2.append(pred2)
        pred2 = torch.stack(preds2, dim=0) # [nt, nz, nx]

        fig2 = shape.plot_compare(pred2)
        fig2.savefig(imagefile2, dpi=300)
    return

def view_shape(resdir, name, blend=True):

    basename = resdir + "shape_" + name
    if blend:
        basename = basename + "_blend"

    imagefile = basename + "_image" + ".png"
    histfile  = basename + "_hist"  + ".png"
    distfile  = basename + "_dist"  + ".png"
    graphfile = basename + "_graph" + ".png"

    if name == "hourglass":
        nw1 = None
        nw2 = None
    elif name == "alldomain":
        nw1 = torch.inf
        nw2 = torch.inf

    # image
    nx, nz, nt = 512, 512, 100
    shape = sandbox.Shape(nx, nz, nt, nw1, nw2, blend=blend)

    fig = shape.plot()
    fig.savefig(imagefile, dpi=300)

    fig = shape.plot_history()
    fig.savefig(histfile, dpi=300)

    fig = shape.plot_distribution()
    fig.savefig(distfile, dpi=300)

    # graph
    nx, nz, nt = 16, 16, 10
    shape = sandbox.Shape(nx, nz, nt, nw1, nw2)
    fig = shape.plot_final_graph()
    fig.savefig(graphfile, dpi=300)

    return
#

if __name__ == "__main__":

    mlutils.set_seed(123)

    parser = argparse.ArgumentParser(description = 'Sandbox')
    parser.add_argument('--gpu_device', default=0, help='GPU device', type=int)
    args = parser.parse_args()

    device = mlutils.select_device()
    if device == "cuda":
        device += f":{args.gpu_device}"

    print(f"using device {device}")

    outdir = "./out/"
    resdir = "./res/"

    # view_shape(resdir, "alldomain")
    # view_shape(resdir, "hourglass")

    # train_cnn(device, outdir, resdir, "hourglass", train=False)

    # train_gnn(device, outdir, resdir, "hourglass", train=False)
    # train_gnn_nextstep(device, outdir, resdir, "hourglass", train=False)

    train_gnn_nextstep(device, outdir, resdir, "hourglass", blend=False, train=True)
#
