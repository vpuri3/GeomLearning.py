#
import torch
import einops
import torch.nn.functional as F

__all__ = [
    'make_optimizer',
    'darcy_deriv_loss',
    #
    'IdentityNormalizer',
    'UnitCubeNormalizer',
    'UnitGaussianNormalizer',
    'TestLoss',
]

#======================================================================#
def make_optimizer(model, lr, weight_decay=0.0):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen weights
        if name.endswith(".bias") or "LayerNorm" in name or "layernorm" in name or "embedding" in name.lower():
            no_decay.append(param)
        elif 'alpha' in name or 'temperature' in name:
            no_decay.append(param)
        elif 'wtq' in name:
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW([
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ], lr=lr)
    
    return optimizer

#======================================================================#
def central_diff(x: torch.Tensor, h: float, resolution: int):
    # assuming PBC
    # x: (batch, n, feats), h is the step size, assuming n = h*w
    x = einops.rearrange(x, 'b (h w) c -> b h w c', h=resolution, w=resolution)
    x = F.pad(x, (0, 0, 1, 1, 1, 1), mode='constant', value=0.)  # [b c t h+2 w+2]
    grad_x = (x[:, 1:-1, 2:, :] - x[:, 1:-1, :-2, :]) / (2 * h)  # f(x+h) - f(x-h) / 2h
    grad_y = (x[:, 2:, 1:-1, :] - x[:, :-2, 1:-1, :]) / (2 * h)  # f(x+h) - f(x-h) / 2h

    return grad_x, grad_y

def darcy_deriv_loss(yh, y, s, dx):
    yh = einops.rearrange(yh, 'b (h w) c -> b c h w', h=s)
    yh = yh[..., 1:-1, 1:-1].contiguous()
    yh = F.pad(yh, (1, 1, 1, 1), "constant", 0)
    yh = einops.rearrange(yh, 'b c h w -> b (h w) c')

    gt_grad_x, gt_grad_y = central_diff(y, dx, s)
    pred_grad_x, pred_grad_y = central_diff(yh, dx, s)
    deriv_loss = F.mse_loss(pred_grad_x, gt_grad_x) + F.mse_loss(pred_grad_y, gt_grad_y)

    return deriv_loss

#======================================================================#
class IdentityNormalizer():
    def __init__(self):
        pass
    
    def to(self, device):
        return self

    def encode(self, x):
        return x

    def decode(self, x):
        return x

#======================================================================#
class UnitCubeNormalizer():
    def __init__(self, X):
        xmin = X[:,:,0].min().item()
        ymin = X[:,:,1].min().item()

        xmax = X[:,:,0].max().item()
        ymax = X[:,:,1].max().item()

        self.min = torch.tensor([xmin, ymin])
        self.max = torch.tensor([xmax, ymax])

    def to(self, device):
        self.min = self.min.to(device)
        self.max = self.max.to(device)

        return self

    def encode(self, x):
        x = (x - self.min) / (self.max - self.min)
        return x

    def decode(self, x):
        return x * (self.max - self.min) + self.min

#======================================================================#
class UnitGaussianNormalizer():
    def __init__(self, X):
        self.mean = X.mean(dim=(0, 1), keepdim=True)
        self.std = X.std(dim=(0, 1), keepdim=True) + 1e-8

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def encode(self, x):
        x = (x - self.mean) / (self.std)
        return x

    def decode(self, x):
        return x * self.std + self.mean

#======================================================================#
class TestLoss():
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(TestLoss, self).__init__()

        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
#======================================================================#
#