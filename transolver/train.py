import numpy as np
import time, json, os
import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader
from tqdm import tqdm


def get_nb_trainable_params(model):
    '''
    Return the number of trainable parameters
    '''
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def train(device, model, train_loader, optimizer, scheduler):
    model.train()

    criterion_func = nn.MSELoss(reduction='none')
    losses = []
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        targets = data.y

        total_loss = criterion_func(out, targets).mean(dim=0)

        total_loss.backward()

        optimizer.step()
        scheduler.step()

        losses.append(total_loss.item())

    return total_loss


@torch.no_grad()
def test(device, model, test_loader):
    model.eval()

    criterion_func = nn.MSELoss(reduction='none')
    losses = []
    for data in test_loader:
        data = data.to(device)
        out = model(data)
        targets = data.y

        loss = criterion_func(out, targets).mean(dim=0)


        losses.append(loss.item())

    return loss


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(device, train_dataset, val_dataset, Net, hparams, path):
    model = Net.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams['lr'],
        total_steps=(len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs'],
        final_div_factor=1000.,
    )
    start = time.time()

    train_loss = 1e5
    train_loss_list = []
    test_loss_list = []
    pbar_train = tqdm(range(hparams['nb_epochs']), position=0)
    for epoch in pbar_train:
        train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, drop_last=True)
        train_loss = train(device, model, train_loader, optimizer, lr_scheduler)
        del (train_loader)

        pbar_train.set_postfix(train_loss=train_loss)
        train_loss_list.append(train_loss.item())
        with open(path + "/train_loss_list.json", 'w') as f:
            json.dump(train_loss_list, f, indent=2) 
        torch.save(model, path + os.sep + 'model.pt')

        test_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=True, drop_last=True)
        test_loss = test(device, model, test_loader)
        test_loss_list.append(test_loss.item())
        with open(path + "/test_loss_list.json", 'w') as f:
            json.dump(test_loss_list, f, indent=2) 

    end = time.time()
    time_elapsed = end - start
    params_model = get_nb_trainable_params(model).astype('float')
    print('Number of parameters:', params_model)
    print('Time elapsed: {0:.2f} seconds'.format(time_elapsed))
    torch.save(model, path + os.sep + 'model.pt')
    with open(path + "/train_loss_list.json", 'w') as f:
        json.dump(train_loss_list, f, indent=2) 
    with open(path + "/test_loss_list.json", 'w') as f:
        json.dump(test_loss_list, f, indent=2) 

    return model
