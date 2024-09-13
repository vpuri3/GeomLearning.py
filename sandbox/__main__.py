#
# 3rd party
import torch
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# builtin
import argparse

# local
import sandbox
import mlutils

class ScalarCNN(nn.Module):
    def __init__(self, shape, w=128, act = nn.ReLU()):
        super().__init__()

        M = shape.final_mask().unsqueeze(0).unsqueeze(0)
        self.register_buffer('M', M)

        # self.cnn = nn.Sequential(
        #     nn.ConvTranspose2d(1, w, kernel_size=4, stride=1, padding=0),
        #     nn.LayerNorm([w, 4, 4]),
        #     act,
        #     nn.ConvTranspose2d(w, w, kernel_size=4, stride=2, padding=1),
        #     nn.LayerNorm([w, 8, 8]),
        #     act,
        #     nn.ConvTranspose2d(w, w, kernel_size=4, stride=2, padding=1),
        #     nn.LayerNorm([w, 16, 16]),
        #     act,
        #     nn.ConvTranspose2d(w, w, kernel_size=4, stride=2, padding=1),
        #     nn.LayerNorm([w, 32, 32]),
        #     act,
        #     nn.ConvTranspose2d(w, w, kernel_size=4, stride=2, padding=1),
        #     nn.LayerNorm([w, 64, 64]),
        #     act,
        #     nn.ConvTranspose2d(w, w, kernel_size=4, stride=2, padding=1),
        #     nn.LayerNorm([w, 128, 128]),
        #     act,
        #     nn.ConvTranspose2d(w, 1, kernel_size=4, stride=2, padding=1),
        # )

        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(1, w, kernel_size=4, stride=1, padding=0), # x4
            nn.LayerNorm([w, 4, 4]),
            act,
            nn.ConvTranspose2d(w, w, kernel_size=8, stride=4, padding=2), # x4
            nn.LayerNorm([w, 16, 16]),
            # act,
            nn.ConvTranspose2d(w, w, kernel_size=8, stride=4, padding=2), # x4
            nn.LayerNorm([w, 64, 64]),
            act,
            nn.ConvTranspose2d(w, 1, kernel_size=8, stride=4, padding=2), # x4
        )

        return

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1) # [N, C, H, W]
        x = self.cnn(x)
        # print(x.shape)
        # assert False
        return x * self.M

def train_scalar_cnn(device):

    modelfile = "./out/" + "scalar_cnn.pth"

    # DATA
    nx, nz, nt = 256, 256, 100
    shape = sandbox.Shape(nx, nz, nt) # hourglass
    shape = sandbox.Shape(nx, nz, nt, nw1=torch.inf, nw2=torch.inf) # alldomain
    data = sandbox.makedata(
        shape, inputs="t", outputs="T", datatype="point-image", mask="finaltime"
    )

    # MODEL
    model = ScalarCNN(shape)
    
    # # TRAIN
    # for lr in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
    #     kw = {
    #         "device" : device,
    #         "lr" : 1e-4,
    #         "_batch_size" : 1,
    #         "nepochs" : 20,
    #     }
    #     trainer = mlutils.Trainer(model, data, **kw)
    #     trainer.train()
    #
    # torch.save(model.to("cpu"), modelfile)

    # VISUALIZE
    model = torch.load(modelfile)#, model)

    fig, axs = plt.subplots(ncols=5, nrows=3, figsize = (15, 9))
    axs[0, 0].set_ylabel(f"True")
    axs[1, 0].set_ylabel(f"Pred")
    axs[2, 0].set_ylabel(f"Errr")

    t, true = data[:]
    pred = model(t)
    errr = torch.abs(true - pred)

    x = shape.x.numpy(force=True)
    z = shape.z.numpy(force=True)

    t = t.numpy(force=True)
    pred = pred.numpy(force=True)
    errr = errr.numpy(force=True)

    it_plt = torch.linspace(0, shape.nt-1, 5)
    it_plt = torch.round(it_plt).to(torch.int).numpy(force=True)
    for (i, it) in enumerate(it_plt):
        axs[0, i].set_title(f"Time {t[it].item():>5f}")

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

    fig.savefig("cnn_compare.png", dpi=300)

    return

def train_mlp_sdf(device):

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

    return

def train_mlp_spacetime_mask(device):

    # DATA
    nx, nz, nt = 256, 256, 50
    shape = sandbox.Shape(nx, nz, nt)
    _data, data_ = sandbox.makedata(
        shape, inputs="xzt", outputs="T", datatype="pointcloud",
        mask="spacetime", split=[.8, .2],
    )

    # MODEL
    width, hidden_dim = 64, 4
    in_dim  = next(iter(_data))[0].shape[0]
    out_dim = next(iter(_data))[1].shape[0]
    model = mlutils.MLP(3, 1, width, hidden_dim, act=nn.Tanh())

    # initialize wandb

    # TRAIN
    kw = {
        "device" : device,
        "lr" : 1e-4,
        "_batch_size" : 512,
        "nepochs" : 20,
    }
    trainer = mlutils.Trainer(model, _data, data_, **kw)
    trainer.train()

    # finalzie wandb

    return

def train_mlp_finaltime_mask(device):

    # DATA
    nx, nz, nt = 256, 256, 50
    shape = sandbox.Shape(nx, nz, nt)
    _data, data_ = sandbox.makedata(
        shape, inputs="xzt", outputs="T", datatype="pointcloud",
        mask="finaltime", split=[.8, .2],
    )

    # MODEL
    width, hidden_dim = 128, 5
    in_dim  = next(iter(_data))[0].shape[0]
    out_dim = next(iter(_data))[1].shape[0]
    model = mlutils.MLP(3, 1, width, hidden_dim, siren=True)

    # TRAIN
    for lr in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
        kw = {
            "device" : device,
            "lr" : lr,
            "_batch_size" : 64,
            "nepochs" : 10,
            "Schedule" : None,
        }
        trainer = mlutils.Trainer(model, _data, data_, **kw)
        trainer.train()

    return

def view_shape():
    nx = 128
    nz = 128
    nt = 20

    fig_kw = {"dpi" : 300,}

    shape = sandbox.Shape(nx, nz, nt)
    fig = shape.plot()
    fig.savefig("hourglass.png", **fig_kw)

    shape = sandbox.Shape(nx, nz, nt, nw1=torch.inf, nw2=torch.inf)
    fig = shape.plot()
    fig.savefig("alldomain.png", **fig_kw)

    return
#

if __name__ == "__main__":

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

    # view_shape()

    # train_mlp_sdf(device)
    # train_mlp_spacetime_mask(device)
    # train_mlp_finaltime_mask(device)

    train_scalar_cnn(device)
#
