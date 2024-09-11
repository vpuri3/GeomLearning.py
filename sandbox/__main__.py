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

def main_train(device):

    insideonly = True
    # insideonly = False

    if insideonly:
        nx, nz, nt = 256, 256, 50 # data
        w, h = 32, 4              # model
        _batch_size = 1024        # trainer
        nepochs = 20
        lr = 5e-4
    else:
        nx, nz, nt = 64, 64, 50 # data
        w, h = 512, 4           # model
        _batch_size = 128       # trainer
        nepochs = 50
        lr = 5e-4
    #

    i, o = 3, 1
    _data, data_ = sandbox.makedata(nx, nz, nt, insideonly=insideonly)
    model = mlutils.MLP(i, o, w, h)

    kw = {
        "device" : device,
        "lr" : lr,
        "_batch_size" : _batch_size,
        "nepochs" : nepochs,
    }

    # initialize wandb

    trainer = mlutils.Trainer(model, _data, data_, **kw)
    trainer.train()

    # finalzie wandb

    return

def main_view():
    nx = 128
    nz = 128
    nt = 20

    shape = sandbox.SandboxShape(nx, nz, nt)
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

    # main_view()
    main_train(device)
#
