#
import torch
from torch import nn

import mlutils

__all__ = [
    'rollout',
]

#======================================================================#
@torch.no_grad()
def rollout(
    model, case_data, transform,
    verbose=False, device=None,
):

    if device is None:
        device = mlutils.select_device(device)
    model.to(device)

    nf = transform.nfields

    eval_data = []
    l2s = []
    r2s = []

    for (istep, data) in enumerate(case_data):
        data = data.clone()
        data.y = transform.makefields(data, istep, scale=True)
        data.e = torch.zeros_like(data.y)
        eval_data.append(data)
        l2s.append(0.)
        r2s.append(1.)

    for k in range(1, len(eval_data)):
        _data = eval_data[k-1].to(device) # given (k-1)-th step
        data  = eval_data[k  ].to(device) # predict k-th step
        
        _data = _data.clone()
        _data.x[:, -nf:] = _data.y[:, -nf:]

        target = transform.makefields(data, k, scale=True)
        data.y = model(_data) + _data.x[:, -nf:]
        data.e = data.y - target

        l2s[k] = nn.MSELoss()(data.e, 0 * data.e).item()
        r2s[k] = mlutils.r2(data.y, target)

        if verbose:
            print(f'Step {k}: {l2s[k], r2s[k]}')

    return eval_data, l2s, r2s

#======================================================================#
