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

def train_sdf(device):
    # parameters
    nx, nz, nt = 64, 64, 50 # data
    w, h = 64, 4           # model
    act = None
    siren = True
    _batch_size = 64        # trainer
    nepochs = 50
    lr = 1e-4
    Schedule = None

    shape = sandbox.Shape(nx, nz, nt)
    _data, data_ = sandbox.makedata(shape, outputs="S")

    i, o = 3, 1
    model = mlutils.MLP(i, o, w, h, act=act, siren=siren)

    return

def train_temp(device):

    # insideonly = True
    insideonly = False

    if insideonly:
        nx, nz, nt = 256, 256, 50 # data
        w, h = 32, 4              # model
        act = nn.Tanh()
        siren = False
        _batch_size = 1024        # trainer
        nepochs = 20
        lr = 5e-4
        Schedule = None
    else:
        nx, nz, nt = 64, 64, 50 # data
        w, h = 512, 4           # model
        act = None
        siren = True
        _batch_size = 64        # trainer
        nepochs = 50
        lr = 1e-4
        Schedule = None
        # Schedule = "OneCycleLR"

    # TODO: label normalization

    shape = sandbox.Shape(nx, nz, nt)
    _data, data_ = sandbox.makedata(shape, insideonly=insideonly, outputs="T")

    i, o = 3, 1
    model = mlutils.MLP(i, o, w, h, act=act, siren=siren)

    kw = {
        "device" : device,
        "lr" : lr,
        "_batch_size" : _batch_size,
        "nepochs" : nepochs,
        "Schedule" : Schedule,
    }

    # initialize wandb

    trainer = mlutils.Trainer(model, _data, data_, **kw)
    trainer.train()

    # finalzie wandb

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

    view_shape()
    # train_temp(device)
    # train_sdf(device)
#
