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

def comparison_plot(shape, pred, true, nt_plt=5):

    fig, axs = plt.subplots(ncols=5, nrows=3, figsize = (15, 9))
    axs[0, 0].set_ylabel(f"True")
    axs[1, 0].set_ylabel(f"Pred")
    axs[2, 0].set_ylabel(f"Errr")

    x = shape.x.numpy(force=True)
    z = shape.z.numpy(force=True)
    t = shape.t.numpy(force=True)

    errr = torch.abs(pred - true)
    pred = pred.numpy(force=True)
    errr = errr.numpy(force=True)

    it_plt = torch.linspace(0, shape.nt-1, nt_plt)
    it_plt = torch.round(it_plt).to(torch.int).numpy(force=True)

    for (i, it) in enumerate(it_plt):
        axs[0, i].set_title(f"Time {t[it].item():>2f}")

        p0 = axs[0, i].contourf(x, z, true[it, 0, :, :], levels=20, cmap='viridis')
        p1 = axs[1, i].contourf(x, z, pred[it, 0, :, :], levels=20, cmap='viridis')
        p2 = axs[2, i].contourf(x, z, errr[it, 0, :, :], levels=20, cmap='viridis')

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

class CNN(nn.Module):
    def __init__(self, shape, w=128, act=nn.ReLU()):
        super().__init__()

        M = shape.final_mask().unsqueeze(0).unsqueeze(0)
        self.register_buffer('M', M)

        self.downsample = nn.Sequential( # [N, 3, 256, 256]
            *mlutils.C2d_block(3, w, None, "2x", act, [w, 256, 256]),
            *mlutils.C2d_block(w, w, None, "2x", act, [w, 128, 128]),
            *mlutils.C2d_block(w, w, None, "2x", act, [w,  64,  64]),
            *mlutils.C2d_block(w, w, None, "2x", act, [w,  32,  32]),
            *mlutils.C2d_block(w, 8, None, "2x", act, [8,  16,  16]),
        )

        self.upsample = nn.Sequential( # [N, 8, 8, 8]
            *mlutils.CT2d_block(8, w, None, "2x", act, [w, 16, 16]),
            *mlutils.CT2d_block(w, w, None, "2x", act, [w, 32, 32]),
            *mlutils.CT2d_block(w, w, None, "2x", act, [w, 64, 64]),
            *mlutils.CT2d_block(w, w, None, "2x", act, [w, 128, 128]),
            *mlutils.CT2d_block(w, 1, None, "2x"),
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.upsample(x)
        return x * self.M
#

class ScalarCNN(nn.Module):
    def __init__(self, shape, w=128, act = nn.ReLU()):
        super().__init__()

        M = shape.final_mask().unsqueeze(0).unsqueeze(0)
        self.register_buffer('M', M)

        # self.mlp = nn.Sequential(
        #     nn.Linear(1, 512), act,
        # )
        # self.cnn = nn.Sequential(
        #     *mlutils.CT2d_block(8, w, None, "2x", act, [w, 16, 16]),
        #     *mlutils.CT2d_block(w, w, None, "2x", act, [w, 32, 32]),
        #     *mlutils.CT2d_block(w, w, None, "2x", act, [w, 64, 64]),
        #     *mlutils.CT2d_block(w, w, None, "2x", act, [w, 128, 128]),
        #     *mlutils.CT2d_block(w, 1, None, "2x"),
        # )

        self.cnn = nn.Sequential(
            *mlutils.CT2d_block(1, w,  4, "1tok", act, [w,  4,  4]),
            *mlutils.CT2d_block(w, w, None, "4x", act, [w, 16, 16]),
            *mlutils.CT2d_block(w, w, None, "4x", act, [w, 64, 64]),
            *mlutils.CT2d_block(w, 1, None, "4x"),
        )

        return

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1) # [N, C, H, W]

        # x = self.mlp(x)
        # x = x.reshape(x.size(0), 8, 8, 8)

        x = self.cnn(x)
        return x * self.M

def train_cnn(device, outdir, resdir, name):

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
    model = CNN(shape)
    
    # TRAIN
    for lr in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
        kw = {
            "device" : device,
            "lr" : lr,
            "_batch_size" : 1,
            "nepochs" : 20,
        }
        trainer = mlutils.Trainer(model, data, **kw)
        trainer.train()

    torch.save(model.to("cpu").state_dict(), modelfile)

    # VISUALIZE
    model = CNN(shape)
    model.load_state_dict(torch.load(modelfile, weights_only=True))

    model.eval()
    t, true = data[:]
    pred = model(t)

    fig = comparison_plot(shape, pred, true)
    fig.savefig(imagefile, dpi=300)

    return

def train_scalar_cnn(device, outdir, resdir, name):

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
    for lr in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
        kw = {
            "device" : device,
            "lr" : 1e-4,
            "_batch_size" : 1,
            "nepochs" : 20,
        }
        trainer = mlutils.Trainer(model, data, **kw)
        trainer.train()
    
    torch.save(model.to("cpu").state_dict(), modelfile)

    # VISUALIZE
    model = ScalarCNN(shape)
    model.load_state_dict(torch.load(modelfile, weights_only=True))

    model.eval()
    t, true = data[:]
    pred = model(t)
    
    fig = comparison_plot(shape, pred, true)
    fig.savefig(imagefile, dpi=300)

    return

def train_mlp_sdf(device, outdir, resdir):

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
    for lr in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
        kw = {
            "device" : device,
            "lr" : lr,
            "_batch_size" : 128,
            "nepochs" : 5,
            "Schedule" : None,
            "lossfun" : nn.L1Loss(),
        }

        trainer = mlutils.Trainer(model, _data, data_, **kw)
        trainer.train()

    torch.save(model.to("cpu").state_dict(), modelfile)

    return

def train_mlp(device, outdir, resdir, name):

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
    for lr in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
        kw = {
            "device" : device,
            "lr" : lr,
            "_batch_size" : 512,
            "nepochs" : 10,
            "Schedule" : None,
        }
        trainer = mlutils.Trainer(model, _data, data_, **kw)
        trainer.train()

    torch.save(model.to("cpu").state_dict(), modelfile)

    return

def view_shape(resdir, name):

    imagefile = resdir + name + ".png"

    if name == "hourglass":
        nw1 = None
        nw2 = None
    elif name == "alldomain":
        nw1 = torch.inf
        nw2 = torch.inf

    nx = 512
    nz = 512
    nt = 20

    fig_kw = {"dpi" : 300,}

    shape = sandbox.Shape(nx, nz, nt, nw1, nw2)
    fig = shape.plot()
    fig.savefig(imagefile, **fig_kw)

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

    # view_shape(resdir, "alldomain")
    # view_shape(resdir, "hourglass")

    # train_mlp_sdf(device, outdir, residr, "hourglass")
    #
    # train_mlp(device, outdir, resdir, "alldomain")
    # train_mlp(device, outdir, resdir, "hourglass")

    # train_cnn(device, outdir, resdir, "alldomain")
    train_cnn(device, outdir, resdir, "hourglass")

    # train_scalar_cnn(device, outdir, resdir, "alldomain")
    # train_scalar_cnn(device, outdir, resdir, "hourglass")
#
