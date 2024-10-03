import torch

import mlutils

__all__ = [
    'prune',
    'split_clone'
]

def prune(model, epoch, densification_interval):
    if epoch % (densification_interval + 1) == 0 and epoch > 0:
        mask_rm = model.mask * (model.alpha < 0.005).view(-1)
        num_rm = mask_rm.sum().item()
        print(f"Pruning {num_rm} Gaussians.")
        model.prune(mask_rm)
    return

def split_clone(model, epoch, densification_interval, gradient_threshold):
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
        print(f"Splitting {len(common_idx)} Gaussians.")
        i0, i1 = model.clone(common_idx)

        scale_factor = 1.6
        model.scale.data[common_idx] /= scale_factor
        model.scale.data[i0:i1     ] /= scale_factor

        # clone points with large coordinate gradients
        # and small gaussian values (i.e. model.scale)
        print(f"Cloning {len(distinct_idx)} Gaussians.")
        i0, i1 = model.clone(distinct_idx)

        step_size = 0.01
        pos_grad = model.mean.grad[distinct_idx]
        pos_grad_mag = torch.norm(pos_grad, dim=1, keepdim=True)
        model.mean.data[i0:i1] += pos_grad / (pos_grad_mag + 1e-6)
    return


def train_full(device, outdir, resdir):

    targetfile  = resdir + "Image-01.png"
    configfile = resdir + "config.yaml"

    # load config
    # with open(configfile, 'r') as config_file:
    #     config = yaml.safe_load(config_file)

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
    target = PIL.Image.open(targetfile)
    target = target.resize(image_size[0:-1])
    target = np.array(target.convert("RGB")) / 255.0
    target = torch.tensor(target).to(torch.float)

    # create model
    model = gsplat2d.GSplat(NG, nG, image_size)

    # move to device
    model  = model.to(device)
    target = target.to(device)

    # optimizer
    opt = optim.Adam(model.parameters(), lr=learning_rate)

    # train loop
    for epoch in range(1, num_epochs+1):

        # pruning
        prune(trainer.model, trainer.epoch, densification_interval)

        # loss computation
        opt.zero_grad()
        image = model()
        loss  = gsplat2d.combined_loss(image, target, lambda_param=0.2)
        loss.backward()

        # densification
        split_clone(trainer.model, trainer.epoch, densification_interval, gradient_threshold)

        # update optimizer
        opt.step()

        # IO
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
#
