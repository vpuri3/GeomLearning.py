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
        trainer.train()

    return model

def train_gnn_nextstep(device, outdir, resdir, name, train=True):

    modelfile  = outdir + "gnn_nextstep_" + name + ".pth"
    graphfile  = resdir + "gnn_nextstep_" + name + "_graph" + ".png"
    imagefile1 = resdir + "gnn_nextstep_" + name + ".png"
    imagefile2 = resdir + "gnn_nextstep_" + name + "_autoregressive" + ".png"

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
    shape = sandbox.Shape(nx, nz, nt, nw1, nw2)
    data = sandbox.makedata(
        shape, inputs="tT", outputs="T", datatype="graph", mask="finaltime",
    )

    # # MODEL
    # ci, co, k = 2, 1, 3
    # model = MaskedUNet(shape, ci, co, k)

    # # MODEL
    # ci, co, k = 2, 1, 3
    # model = MaskedUNet(shape, ci, co, k)
    #
    # # TRAIN
    # if train:
    #     train_loop(model, data, device=device, _batch_size=1)
    #     torch.save(model.to("cpu").state_dict(), modelfile)
    #
    # # VISUALIZE
    # model.eval()
    # model.load_state_dict(torch.load(modelfile, weights_only=True))
    #
    # with torch.no_grad():
    #     xztT, _ = data[:]
    #
    #     pred1 = (model(xztT) + xztT[:, -1].unsqueeze(1)).squeeze(1)
    #     pred1 = torch.cat([xztT[0, -1].unsqueeze(0), pred1], dim=0)
    #
    #     fig1 = shape.comparison_plot(pred1)
    #     fig1.savefig(imagefile1, dpi=300)
    #
    #     preds2 = []
    #     preds2.append(xztT[0, -1].unsqueeze(0))
    #     for i in range(xztT.shape[0]):
    #         pred2 = preds2[-1]
    #         pred2 = pred2 + model(xztT[i].unsqueeze(0)).squeeze(1)
    #         preds2.append(pred2)
    #     pred2 = torch.cat(preds2, dim=0)
    #
    #     fig2 = shape.comparison_plot(pred2)
    #     fig2.savefig(imagefile2, dpi=300)

    return

class MaskedUNet(nn.Module):
    def __init__(self, shape, ci, co, k):
        super().__init__()

        M = shape.final_mask().unsqueeze(0).unsqueeze(0)
        self.register_buffer('M', M)
        self.unet = mlutils.UNet(ci, co, k)
        return

    def forward(self, x):
        x = self.unet(x)
        return x * self.M
#

def train_cnn_nextstep(device, outdir, resdir, name, train=True):

    modelfile  = outdir + "cnn_nextstep_" + name + ".pth"
    imagefile1 = resdir + "cnn_nextstep_" + name + ".png"
    imagefile2 = resdir + "cnn_nextstep_" + name + "_autoregressive" + ".png"

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
        shape, inputs="tT", outputs="T", datatype="image-nextstep",
        mask="finaltime",
    )

    # MODEL
    ci, co, k = 2, 1, 3
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

        pred1 = (model(xztT) + xztT[:, -1].unsqueeze(1)).squeeze(1)
        pred1 = torch.cat([xztT[0, -1].unsqueeze(0), pred1], dim=0)

        fig1 = shape.comparison_plot(pred1)
        fig1.savefig(imagefile1, dpi=300)

        preds2 = []
        preds2.append(xztT[0, -1].unsqueeze(0))
        for i in range(xztT.shape[0]):
            pred2 = preds2[-1]
            pred2 = pred2 + model(xztT[i].unsqueeze(0)).squeeze(1)
            preds2.append(pred2)
        pred2 = torch.cat(preds2, dim=0)

        fig2 = shape.comparison_plot(pred2)
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

    fig = shape.comparison_plot(pred)
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
    
    fig = shape.comparison_plot(pred)
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
        train_loop(model, data, device=device, _batch_size=128, lossfun=nn.L1Loss())
        torch.save(model.to("cpu").state_dict(), modelfile)

    # VISUALIZE

    return

def train_mlp(device, outdir, resdir, name, train=True):

    modelfile = outdir + "mlp" + name + ".pth"
    imagefile = resdir + "mlp" + name + ".png"

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

    fig = shape.comparison_plot(pred)
    fig.savefig(imagefile, dpi=300)

    return

def view_shape(resdir, name):

    imagefile = resdir + "image_" + name + ".png"
    graphfile = resdir + "graph_" + name + ".png"

    if name == "hourglass":
        nw1 = None
        nw2 = None
    elif name == "alldomain":
        nw1 = torch.inf
        nw2 = torch.inf

    # image
    nx, nz, nt = 512, 512, 10
    shape = sandbox.Shape(nx, nz, nt, nw1, nw2)
    fig = shape.plot()
    fig.savefig(imagefile, dpi=300)

    # graph
    nx, nz, nt = 32, 32, 10
    shape = sandbox.Shape(nx, nz, nt, nw1, nw2)
    fig = shape.plot_final_graph()
    fig.savefig(graphfile, dpi=300)

    return
#

if __name__ == "__main__":

    # random seed
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser(description = 'Sandbox')
    parser.add_argument('--gpu_device', default=0, help='GPU device', type=int)
    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    if device == "cuda":
        device += f":{args.gpu_device}"

    print(f"using device {device}")

    outdir = "./out/"
    resdir = "./res/"

    view_shape(resdir, "alldomain")
    view_shape(resdir, "hourglass")

    # train_mlp_sdf(device, outdir, residr, "hourglass")

    # train_mlp(device, outdir, resdir, "alldomain")
    # train_mlp(device, outdir, resdir, "hourglass")

    # train_cnn(device, outdir, resdir, "alldomain")
    # train_cnn(device, outdir, resdir, "hourglass")

    # train_gnn_nextstep(device, outdir, resdir, "alldomain")
    # train_gnn_nextstep(device, outdir, resdir, "hourglass")

    # train_cnn_nextstep(device, outdir, resdir, "alldomain", train=False)
    # train_cnn_nextstep(device, outdir, resdir, "hourglass", train=False)

    # train_scalar_cnn(device, outdir, resdir, "alldomain")
    # train_scalar_cnn(device, outdir, resdir, "hourglass")
#
