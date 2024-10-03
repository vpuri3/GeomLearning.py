#
# https://github.com/OutofAi/2D-Gaussian-Splatting/
#
import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import PIL

# local
import gsplat2d
import mlutils

# builtin
import os
import gc
import time
import argparse
import requests

def test_raster(device, outdir):

    # scale_x = torch.tensor([1.0, 0.5, 0.5], device=device)
    # scale_y = torch.tensor([1.0, 0.5, 1.5], device=device)
    # scale = torch.stack([scale_x, scale_y], dim=-1)
    # rotation = torch.tensor([0.0, 0.0, -0.5], device=device) # a value between -pi/2 and pi/2
    # means = torch.tensor([(-0.5, -0.5), (0.8, 0.8), (0.5, 0.5)], device=device)
    # colors  = torch.tensor([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)], device=device)
    # image = gsplat2d.rasterize(scale, rotation, means, colors)

    model = gsplat2d.GSplat(15, 15).to(device)
    model = gsplat2d.GSplat(100).to(device)
    image = model()

    fig, ax = plt.subplots(1, 1)
    ax.imshow(image.numpy(force=True))
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(outdir + "raster.png", dpi=300)

    return

def download_config(outdir):
    url1 = 'https://raw.githubusercontent.com/OutofAi/2D-Gaussian-Splatting/main/Image-01.png'
    filename1 = outdir + url1.split('/')[-1]
    response1 = requests.get(url1)
    with open(filename1, 'wb') as f:
        f.write(response1.content)

    url2 = 'https://raw.githubusercontent.com/OutofAi/2D-Gaussian-Splatting/main/config.yml'
    filename2 = outdir + url2.split('/')[-1]
    response2 = requests.get(url2)
    with open(filename2, 'wb') as f:
        f.write(response2.content)
    return filename1, filename2

def train(device, outdir):

    imagefile  = outdir + "Image-01.png"

    # hyper-parameters
    nG = 200
    NG = 500
    image_size = (128, 128, 3)
    num_epochs = 2000
    learning_rate = 0.001
    densification_interval = 100
    gradient_threshold = 0.002
    gauss_threshold = 0.75

    # get target image
    target = PIL.Image.open(imagefile)
    target = target.resize(image_size[0:-1])
    target = np.array(target.convert("RGB")) / 255.0
    target = torch.tensor(target).to(torch.float)

    # create model
    model = gsplat2d.GSplat(NG, nG, image_size)

    # create dataset
    data = [(0, target)]
    def collate_fn(batch):
        batch = torch.utils.data._utils.collate.default_collate(batch)
        return batch[0], batch[1].squeeze(0)

    loader = torch.utils.data.DataLoader(data, collate_fn=collate_fn)

    # create trainer
    trainer = mlutils.Trainer(
        model, data, device=device, nepochs=2000, lr=learning_rate,
        collate_fn=collate_fn,
        stats_every=20, print_config=False, lossfun=gsplat2d.combined_loss,
    )

    def cb_prune(trainer: mlutils.Trainer):
        return gsplat2d.prune(trainer.model, trainer.epoch, densification_interval)

    def cb_split_clone(trainer: mlutils.Trainer):
        return gsplat2d.split_clone(trainer.model, trainer.epoch, densification_interval, gradient_threshold)

    # call back functions for densification
    trainer.add_callback("batch_start", cb_prune)
    trainer.add_callback("batch_post_grad", cb_split_clone)

    # train model
    trainer.train()

    # save output image
    image = model()
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image.numpy(force=True))
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(outdir + "out.png", dpi=300)

    return

if __name__ == "__main__":
    mlutils.set_seed(123)

    parser = argparse.ArgumentParser(description = 'GSplat2D')
    parser.add_argument('--gpu_device', default=0, help='GPU device', type=int)
    args = parser.parse_args()

    device = mlutils.select_device()
    if device == "cuda":
        device += f":{args.gpu_device}"

    print(f"using device {device}")

    outdir = "./gsplat2d/"
    # test_raster(device, outdir)
    # imagefile, configfile = download_config(outdir)
    train(device, outdir)

    pass
#
