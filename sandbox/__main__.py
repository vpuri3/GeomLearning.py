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

class SandboxCNN:
    def __init__(self, in_dim, out_dim,):
        super().__init__()
        pass

    def forward(self, x):
        pass

def train_cnn(device):
    mask = "finaltime"
    datatype = "image"

    nx, nz, nt = 256, 256, 100
    shape = sandbox.Shape(nx, nz, nt)
    data = sandbox.makedata(shape, inputs="xzt", outputs="T", datatype=datatype, mask=mask)

    assert False

    lr = 5e-4
    _batch_size = 1
    neochs = 50

    return

def train_mlp_sdf(device):

    sdf_clamp = 1e-2

    # DATA
    nx, nz, nt = 128, 128, 50
    shape = sandbox.Shape(nx, nz, nt)
    _data, data_ = sandbox.makedata(
        shape, inputs="xzt", outputs="S", datatype="pointcloud",
        mask=None, sdf_clamp=sdf_clamp,
    )
    
    # MODEL
    width, hidden_dim = 128, 5
    in_dim  = next(iter(_data))[0].shape[0]
    out_dim = next(iter(_data))[1].shape[0]
    model = mlutils.MLP(3, 1, width, hidden_dim, siren=True)
    model = nn.Sequential(*model, mlutils.SDFClamp(sdf_clamp))

    # TRAIN
    for lr in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
        kw = {
            "device" : device,
            "lr" : lr,
            "_batch_size" : 64,
            "nepochs" : 2,
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
        mask="spacetime",
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
    _data, data_ = sandbox.makedata(shape, inputs="xzt", outputs="T",
        datatype="pointcloud", mask="finaltime",)

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

    shape = sandbox.Shape(nx, nz, nt)
    shape.plot()

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

    train_mlp_sdf(device)
    # train_mlp_spacetime_mask(device)
    # train_mlp_finaltime_mask(device)

    # train_cnn(device)
#
