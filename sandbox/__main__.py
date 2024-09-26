#
# 3rd party
import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

# builtin
import random
import argparse

# local
import sandbox
import mlutils

def train_loop(model, data, lrs=None, nepochs=None, **kw):

    if lrs is None:
        lrs = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]

    if nepochs is None:
        nepochs = [2, 5, 10, 10, 10]

    assert len(lrs) == len(nepochs)

    for i in range(len(lrs)):
        kw = {
            **kw,
            "lr" : lrs[i],
            "nepochs" : nepochs[i],
        }
        trainer = mlutils.Trainer(model, data, **kw)
        # def cb_epoch(trainer: mlutils.Trainer):
        #     print(f"epoch callback")
        # trainer.add_callback("epoch_end", cb_epoch)
        trainer.train()

    return model

def train_gnn_nextstep(device, outdir, resdir, name, blend=False, train=True):

    outname = outdir + "gnn_nextstep_" + name
    resname = resdir + "gnn_nextstep_" + name

    if blend:
        outname = outname + "_blend"
        resname = resname + "_blend"

    modelfile  = outname + ".pth"
    graphfile  = resname + "_graph" + ".png"
    imagefile1 = resname + ".png"
    imagefile2 = resname + "_autoregressive" + ".png"

    # DATA
    if name == "hourglass":
        nw1 = None
        nw2 = None
    elif name == "alldomain":
        nw1 = torch.inf
        nw2 = torch.inf

    nx, nz, nt = 32, 32, 100
    shape = sandbox.Shape(nx, nz, nt, nw1, nw2)
    fig = shape.plot_final_graph()
    fig.savefig(graphfile, dpi=300)

    # nx, nz, nt = 128, 128, 100
    # shape = sandbox.Shape(nx, nz, nt, nw1, nw2)
    data = sandbox.makedata(
        shape, inputs="tT", outputs="T", datatype="graph", mask="finaltime",
    )

    # # MODEL
    # ci, co, k = 2, 1, 3
    # model = MaskedUNet(shape, ci, co, k)

    return

class MaskedUNet(nn.Module):
    def __init__(self, shape, ci, co, k):
        super().__init__()

        self.shape = shape
        self.unet = mlutils.UNet(ci, co, k)

        # M = shape.final_mask().unsqueeze(0).unsqueeze(0)
        # self.register_buffer('M', M)

    @torch.no_grad()
    def compute_mask(self, x):
        x0, z0, t0 = [x[0,c,:,:] for c in range(3)]
        t1 = t0 + self.shape.dt()
        M = self.shape.mask(x0, z0, t1)

        # M = self.shape.mask(x0, z0, torch.ones_like(t0))

        return M.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        y = self.unet(x)
        M = self.compute_mask(x)

        return y * M
#

def train_cnn_nextstep(device, outdir, resdir, name, blend=False, train=True):

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

    nx, nz, nt = 256, 256, 100
    shape = sandbox.Shape(nx, nz, nt, nw1, nw2, blend=blend)
    data = sandbox.makedata(
        shape, inputs="xztT", outputs="T", datatype="image-nextstep",
        mask="finaltime",
    )

    # MODEL
    ci, co, k = 4, 1, 3
    model = MaskedUNet(shape, ci, co, k)

    # TRAIN
    if train:
        train_loop(model, data, device=device, _batch_size=1)
        torch.save(model.to("cpu").state_dict(), modelfile)

    # VISUALIZE
    model.eval()
    model.load_state_dict(torch.load(modelfile, weights_only=True))

    with torch.no_grad():
        xztT, _ = data[:]

        pred1 = mlutils.eval_model(xztT, model, device, batch_size=4)
        pred1 = (pred1 + xztT[:, -1].unsqueeze(1)).squeeze(1)
        pred1 = torch.cat([xztT[0, -1].unsqueeze(0), pred1], dim=0)

        def process(y0, y1):
            xzt = y0[:, 0:3, :, :]
            temp = y0[:, -1, :, :].unsqueeze(1) + y1
            return torch.cat([xzt, temp], dim=1)

        def save(ys):
            temps = [y[:, -1, :, :] for y in ys]
            return torch.cat(temps, dim=0)

        # begin rollout from timestep 10

        # pred2 = mlutils.autoregressive_rollout(
        #     xztT[0].unsqueeze(0), model, len(data),
        #     process=process, save=save, device=device,
        # )

        preds2 = []
        K = 10
        for k in range(K):
            preds2.append(xztT[k, -1].unsqueeze(0))
        for k in range(K-1, xztT.shape[0]):
            pred2 = preds2[-1]
            pred2 = pred2 + model(xztT[k].unsqueeze(0)).squeeze(1)
            preds2.append(pred2)
        pred2 = torch.cat(preds2, dim=0)

        fig1 = shape.plot_compare(pred1)
        fig1.savefig(imagefile1, dpi=300)

        fig2 = shape.plot_compare(pred2)
        fig2.savefig(imagefile2, dpi=300)

    return

def train_cnn(device, outdir, resdir, name, train=True):

    modelfile = outdir + "cnn_" + name + ".pth"
    imagefile = resdir + "cnn_" + name + ".png"

    # DATA
    if name == "hourglass":
        nw1 = None
        nw2 = None
    elif name == "alldomain":
        nw1 = torch.inf
        nw2 = torch.inf

    nx, nz, nt = 256, 256, 100
    shape = sandbox.Shape(nx, nz, nt, nw1, nw2)
    data = sandbox.makedata(
        shape, inputs="xzt", outputs="T", datatype="image", mask="finaltime"
    )

    # MODEL
    ci, co, k = 3, 1, 3
    model = MaskedUNet(shape, ci, co, k)

    # TRAIN
    if train:
        train_loop(model, data, device=device, _batch_size=1)
        torch.save(model.to("cpu").state_dict(), modelfile)

    # VISUALIZE
    model.load_state_dict(torch.load(modelfile, weights_only=True))

    model.eval()
    xzt, _ = data[:]
    pred = model(xzt).squeeze(1)

    fig = shape.plot_compare(pred)
    fig.savefig(imagefile, dpi=300)

    return

class ScalarCNN(nn.Module):
    def __init__(self, shape, w=128, act = nn.ReLU()):
        super().__init__()

        M = shape.final_mask().unsqueeze(0).unsqueeze(0)
        self.register_buffer('M', M)

        self.cnn = nn.Sequential(
            *mlutils.CT2d_block(1, w,  4, "1tok", act, [w,  4,  4]),
            *mlutils.CT2d_block(w, w, None, "4x", act, [w, 16, 16]),
            *mlutils.CT2d_block(w, w, None, "4x", act, [w, 64, 64]),
            *mlutils.CT2d_block(w, 1, None, "4x"),
        )

        return

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1) # [N, C] --> [N, C, H, W]
        x = self.cnn(x)
        return x * self.M

def train_scalar_cnn(device, outdir, resdir, name, train=True):

    modelfile = outdir + "scalar_cnn_" + name + ".pth"
    imagefile = resdir + "scalar_cnn_" + name + ".png"

    # DATA

    if name == "hourglass":
        nw1 = None
        nw2 = None
    elif name == "alldomain":
        nw1 = torch.inf
        nw2 = torch.inf

    nx, nz, nt = 256, 256, 100
    shape = sandbox.Shape(nx, nz, nt, nw1,  nw2)
    data = sandbox.makedata(
        shape, inputs="t", outputs="T", datatype="point-image", mask="finaltime"
    )

    # MODEL
    model = ScalarCNN(shape)
    
    # TRAIN
    if train:
        train_loop(model, data, device=device, _batch_size=1)
        torch.save(model.to("cpu").state_dict(), modelfile)

    # VISUALIZE
    model = ScalarCNN(shape)
    model.load_state_dict(torch.load(modelfile, weights_only=True))

    model.eval()
    t, _ = data[:]
    pred = model(t).squeeze(1)
    
    fig = shape.plot_compare(pred)
    fig.savefig(imagefile, dpi=300)

    return

def train_mlp_sdf(device, outdir, resdir, train=True):

    modelfile = outdir + "mlp_sdf" + ".pth"
    imagefile = resdir + "mlp_sdf" + ".png"

    # DATA
    sdf_clamp = 1e-2
    nx, nz, nt = 128, 128, 50
    shape = sandbox.Shape(nx, nz, nt)
    _data, data_ = sandbox.makedata(
        shape, inputs="xzt", outputs="S", datatype="pointcloud",
        mask=None, sdf_clamp=sdf_clamp, split=[.8, .2],
    )

    # MODEL
    width, hidden_dim = 512, 5
    in_dim  = next(iter(_data))[0].shape[0]
    out_dim = next(iter(_data))[1].shape[0]
    model = mlutils.MLP(3, 1, width, hidden_dim, siren=True)
    model = nn.Sequential(*model, mlutils.SDFClamp(sdf_clamp))

    # TRAIN
    if train:
        train_loop(model, _data, data_=data_, device=device, _batch_size=128, lossfun=nn.L1Loss())
        torch.save(model.to("cpu").state_dict(), modelfile)

    # VISUALIZE

    return

def train_mlp(device, outdir, resdir, name, train=True):

    modelfile = outdir + "mlp_" + name + ".pth"
    imagefile = resdir + "mlp_" + name + ".png"

    # DATA

    if name == "hourglass":
        nw1 = None
        nw2 = None
    elif name == "alldomain":
        nw1 = torch.inf
        nw2 = torch.inf

    nx, nz, nt = 128, 128, 50
    shape = sandbox.Shape(nx, nz, nt, nw1, nw2)
    _data, data_ = sandbox.makedata(
        shape, inputs="xzt", outputs="T", datatype="pointcloud",
        mask="finaltime", split=[.8, .2],
    )

    # MODEL
    width, hidden_dim = 128, 5
    in_dim  = next(iter(_data))[0].shape[0]
    out_dim = next(iter(_data))[1].shape[0]
    model = mlutils.MLP(in_dim, out_dim, width, hidden_dim, siren=True)

    # TRAIN
    if train:
        train_loop(model, data, device=device, _batch_size=512)
        torch.save(model.to("cpu").state_dict(), modelfile)

    # VISUALIZE
    model.load_state_dict(torch.load(modelfile, weights_only=True))

    (x, z, t), (mask, _, _, _) = shape.fields_dense()
    xzt  = torch.stack([x, z, t], dim=3)
    pred = model(xzt).squeeze(-1) * mask[-1].unsqueeze(0)

    fig = shape.plot_compare(pred)
    fig.savefig(imagefile, dpi=300)

    return

def view_shape(resdir, name, blend=False):

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
    # view_shape(resdir, "alldomain", blend=True)
    #
    # view_shape(resdir, "hourglass")
    # view_shape(resdir, "hourglass", blend=True)

    # train_mlp_sdf(device, outdir, resdir, "hourglass")

    # train_mlp(device, outdir, resdir, "alldomain")
    # train_mlp(device, outdir, resdir, "hourglass")

    # train_gnn_nextstep(device, outdir, resdir, "alldomain")
    # train_gnn_nextstep(device, outdir, resdir, "hourglass")

    # train_cnn_nextstep(device, outdir, resdir, "alldomain", train=False)
    # train_cnn_nextstep(device, outdir, resdir, "hourglass", train=False)

    # train_cnn_nextstep(device, outdir, resdir, "alldomain", blend=True, train=False)
    train_cnn_nextstep(device, outdir, resdir, "hourglass", blend=True, train=False)

    # train_cnn(device, outdir, resdir, "alldomain")
    # train_cnn(device, outdir, resdir, "hourglass")

    # train_scalar_cnn(device, outdir, resdir, "alldomain")
    # train_scalar_cnn(device, outdir, resdir, "hourglass")
#
