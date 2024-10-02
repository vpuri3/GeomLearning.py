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

def test_raster(device, outdir, resdir):

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
    fig.savefig(resdir + "raster.png", dpi=300)

    return

def download_config(outdir, resdir):
    url1 = 'https://raw.githubusercontent.com/OutofAi/2D-Gaussian-Splatting/main/Image-01.png'
    filename1 = resdir + url1.split('/')[-1]
    response1 = requests.get(url1)
    with open(filename1, 'wb') as f:
        f.write(response1.content)

    url2 = 'https://raw.githubusercontent.com/OutofAi/2D-Gaussian-Splatting/main/config.yml'
    filename2 = resdir + url2.split('/')[-1]
    response2 = requests.get(url2)
    with open(filename2, 'wb') as f:
        f.write(response2.content)
    return filename1, filename2

def train(device, outdir, resdir):

    imagefile  = resdir + "Image-01.png"
    configfile = resdir + "config.yaml"

    # load config
    # with open(configfile, 'r') as config_file:
    #     config = yaml.safe_load(config_file)
    
    # Extract values from the loaded config
    config = {
        "KERNEL_SIZE"            : 101,           # config["KERNEL_SIZE"]
        "image_size"             : (128, 128, 3), # (256, 256, 3), # tuple(config["image_size"])
        "primary_gaussians"      : 100, # 1000,          # config["primary_samples"]
        "backup_gaussians"       : 400, # 4000,          # config["backup_samples"]
        "num_epochs"             : 2000,          # config["num_epochs"]
        "densification_interval" : 100, #300,           # config["densification_interval"]
        "learning_rate"          : 0.001,         # config["learning_rate"]
        "display_interval"       : 100,           # config["display_interval"]
        "gradient_threshold"     : 0.002,         # config["gradient_threshold"]
        "gauss_threshold"        : 0.75,          # config["gaussian_threshold"]
        "display_loss"           : False,         # config["display_loss"]
    }

    nG = config["primary_gaussians"]
    NG = nG + config["backup_gaussians"]
    image_size = config["image_size"]

    num_epochs = config["num_epochs"]
    learning_rate = config["learning_rate"]
    densification_interval = config["densification_interval"]
    gradient_threshold = config["gradient_threshold"]
    gauss_threshold = config["gauss_threshold"]

    # get target image
    target = PIL.Image.open(imagefile)
    target = target.resize(config["image_size"][0:-1])
    target = np.array(target.convert("RGB")) / 255.0
    target = torch.tensor(target).to(torch.float)

    # initialize GSplat
    model = gsplat2d.GSplat(NG, nG, image_size)

    # move to device
    model  = model.to(device)
    target = target.to(device)

    loss_hist = []
    opt = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs+1):

        # pruning
        if epoch % (densification_interval + 1) == 0 and epoch > 0:
            mask_rm = model.mask * (model.alpha < 0.01).view(-1)
            num_rm = mask_rm.sum().item()
            print(f"Pruning {num_rm} Gaussians.")
            model.prune(mask_rm)

        # loss computation
        opt.zero_grad()
        image = model()
        loss  = gsplat2d.combined_loss(image, target, lambda_param=0.2)
        loss.backward()

        # densification
        if epoch % densification_interval == 0 and epoch > 0:

            # gradient norm of position
            grad_mean = torch.norm(model.mean.grad[model.mask], dim=1, p=2)

            # covariance (S matrix)
            scale_val = torch.norm(torch.sigmoid(model.scale[model.mask]), dim=1, p=2)

            mean_sort , mean_idx_sort  = torch.sort(grad_mean, descending=True)
            scale_sort, scale_idx_sort = torch.sort(scale_val, descending=True)

            mask_mean  = mean_sort  > gradient_threshold
            mask_scale = scale_sort > gradient_threshold

            idx_mean  = mean_idx_sort[mask_mean]
            idx_scale = scale_idx_sort[mask_scale]

            common_idx_mask = torch.isin(idx_mean, idx_scale)
            common_idx   = idx_mean[ common_idx_mask]
            distinct_idx = idx_mean[~common_idx_mask]

            # split points with large coordinate gradients
            # and large gaussian values (i.e. model.scale)
            if len(common_idx) > 0:
                print(f"Splitting {len(common_idx)} Gaussians.")
                model.split(common_idx)

            # clone points with large coordinate gradients
            # and small gaussian values (i.e. model.scale)
            print(f"Cloning {len(distinct_idx)} Gaussians.")
            model.clone(distinct_idx)

        # update optimizer
        opt.step()

        # IO
        loss_hist.append(loss.item())
        if epoch % 20 == 0:
            print(
                f"Epoch [{epoch} / {num_epochs}]: " +
                f"NG: {model.active_gaussians()}, " +
                f"LOSS: {loss.item()}"
            )
    # endfor

    image = model()
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image.numpy(force=True))
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(resdir + "out.png", dpi=300)

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

    outdir = "./out-gsplat/"
    resdir = "./res-gsplat/"

    # test_raster(device, outdir, resdir)
    # imagefile, configfile = download_config(outdir, resdir)
    train(device, outdir, resdir)

    pass
#
