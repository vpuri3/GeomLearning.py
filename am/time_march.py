#
import torch
from torch import nn

import mlutils

__all__ = [
    'march_case',
]

#======================================================================#
@torch.no_grad()
def march_case(model, case_data, transform,
        autoreg=True, K=1, verbose=True, device=None,
):

    if device is None:
        # TODO: make select_device work with torchrun
        device = mlutils.select_device(device)

    model.to(device)

    scale = []
    scale = [*scale, transform.disp_scale ] if transform.disp  else scale
    scale = [*scale, transform.vmstr_scale] if transform.vmstr else scale
    scale = [*scale, transform.temp_scale ] if transform.temp  else scale
    scale = torch.tensor(scale)

    nf = transform.disp + transform.vmstr + transform.temp 
    assert nf == len(scale)

    def makefields(data):
        xs = []
        xs = [*xs, data.disp[:,2].view(-1,1)] if transform.disp  else xs
        xs = [*xs, data.vmstr.view(-1,1)    ] if transform.vmstr else xs
        xs = [*xs, data.temp.view(-1,1)     ] if transform.temp  else xs

        return torch.cat(xs, dim=-1) / scale.to(xs[0].device)

    eval_data = []
    for data in case_data:
        data = data.clone()
        data.y = makefields(data)
        data.e = torch.zeros_like(data.y)
        eval_data.append(data)

    for k in range(K, len(eval_data)):
        _data = eval_data[k-1].to(device) # given (k-1)-th step
        data  = eval_data[k  ].to(device) # predict k-th step

        if autoreg:
            _data = _data.clone()
            _data.x[:, -nf:] = _data.y[:, -nf:]

        target = makefields(data)

        data.y = model(_data) + _data.x[:, -nf:]
        data.e = data.y - target

        if verbose:
            l1 = nn.L1Loss()( data.e, 0 * data.e).item()
            l2 = nn.MSELoss()(data.e, 0 * data.e).item()
            r2 = mlutils.r2(data.y, target)
            print(f'Step {k}: {l1, l2, r2}')

    return eval_data

#======================================================================#
