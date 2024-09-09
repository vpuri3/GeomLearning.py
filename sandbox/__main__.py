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

    # data
    Nx = 128
    Nz = 128
    _data, data_ = sandbox.makedata(Nx, Nz)

    # model
    i, o =  2, 2
    w, h = 32, 2
    model = mlutils.MLP(i, o, w, h)

    # initialize wandb

    trainer = mlutils.Trainer(model, _data, data_, device)
    trainer.train()

    # finalzie wandb

    return

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
