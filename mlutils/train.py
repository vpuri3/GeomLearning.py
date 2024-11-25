#
# 3rd party
import torch
from torch import nn, optim
from torch import distributed as dist
import torch_geometric as pyg

from tqdm import tqdm

# builtin
import os
import math
import time
import collections

# local
from mlutils.utils import (
    num_parameters, select_device, is_torchrun, 
)

__all__ = [
    'Trainer',
]

class Trainer:
    def __init__(
        self, 
        model,
        _data,
        data_=None,

        GNN=False,
        device=None,

        collate_fn=None,
        _batch_size=None,
        batch_size_=None,
        _batch_size_=None,

        lr=None,
        weight_decay=None,

        Opt=None,
        Schedule=None,

        lossfun=None,
        nepochs=None,

        statsfun=None,
        verbose=True,
        print_config=False,
        print_batch=True,
        print_epoch=True,
        stats_every=1, # stats every k epochs
    ):

        ###
        # DEVICE
        ###

        self.DISTRIBUTED = is_torchrun()
        self.LOCAL_RANK = int(os.environ['LOCAL_RANK']) if self.DISTRIBUTED else 0
        if self.DISTRIBUTED:
            assert dist.is_initialized()
            self.DDP = dist.get_world_size() > 1
            self.device = self.LOCAL_RANK
        else:
            self.DDP = False
            self.device = select_device(device, verbose=True)

        ###
        # DATA
        ###

        if _data is None:
            raise ValueError('_data passed to Trainer cannot be None.')

        if _batch_size is None:
            _batch_size = 32
        if _batch_size_ is None:
            _batch_size_ = len(_data)
        if data_ is not None:
            if batch_size_ is None:
                batch_size_ = len(data_)

        ###
        # MODEL
        ###

        if verbose and (self.LOCAL_RANK == 0):
            print(f"Moving model with {num_parameters(model)} parameters to device {device}")
        model.to(device)

        if self.DDP:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[device])

        ###
        # OPTIMIZER
        ###

        param = model.parameters()

        if lr is None:
            lr = 1e-3
        if weight_decay is None:
            weight_decay = 0.0

        if Opt == "Adam" or Opt is None:
            opt = optim.Adam(param, lr=lr)
        elif Opt == "AdamW":
            opt = optim.AdamW(param, lr=lr, weight_decay=weight_decay)
        else:
            raise NotImplementedError()

        if lossfun is None:
            lossfun = nn.MSELoss()
        if nepochs is None:
            nepochs = 100

        if Schedule == "OneCycleLR":
            niters = nepochs * len(_loader)
            schedule = optim.lr_scheduler.OneCycleLR(opt, 1e-2, total_steps=niters)
        elif Schedule == "CosineAnnealingLR":
            niters = nepochs * len(_loader)
            schedule = optim.lr_scheduler.CosineAnnealingLR(opt, niters)
        elif Schedule is None:
            schedule = optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=1e10)
        else:
            raise NotImplementedError()

        config = {
            "GNN" : GNN,
            "device" : device,

            "data_size" : len(_data),
            "num_batches" : len(_data) // _batch_size,
            "batch_size" : _batch_size,

            "num_parameters" : num_parameters(model),

            "learning_rate" : lr,
            "weight_decay" : weight_decay,
            # "optimizer" : str(opt),
            "schedule"  : str(schedule),

            "nepochs" : nepochs,
            "lossfun" : str(lossfun),
        }

        if verbose and print_config and (self.LOCAL_RANK == 0):
            print(model)
            print(f"Trainer config:")
            for (k, v) in config.items():
                print(f"{k} : {v}")

        # ASSIGN TO SELF

        # MISC
        self.GNN = GNN

        # DATA
        self._data = _data
        self.data_ = data_
        self.collate_fn = collate_fn
        self._batch_size = _batch_size
        self.batch_size_ = batch_size_
        self._batch_size_ = _batch_size_

        # MODEL
        self.model = model

        # OPT
        self.opt = opt
        self.schedule = schedule

        self.lossfun = lossfun
        self.nepochs = nepochs

        # config, callback and printing
        self.config = config
        self.statsfun = statsfun
        self.callbacks = collections.defaultdict(list)

        self.verbose = verbose
        self.print_config = print_config
        self.print_batch = print_batch
        self.print_epoch = print_epoch
        self.stats_every = stats_every

        # iteration
        self.epoch = 0

    # https://github.com/karpathy/minGPT/
    def add_callback(self, event: str, callback):
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        self.callbacks[event] = [callback]

    def trigger_callbacks(self, event: str):
        for callback in self.callbacks[event]:
            callback(self)

    def save(self, save_path: str): # call only if device==0
        if self.LOCAL_RANK != 0:
            return

        snapshot = dict()
        snapshot['epoch'] = self.epoch
        if self.DDP:
            snapshot['model_state'] = self.model.module.state_dict()
        else:
            snapshot['model_state'] = self.model.state_dict()
        snapshot['opt'] = self.opt
        torch.save(snapshot, save_path)

        return

    def load(self, load_path: str):
        print(f"Loading {load_path}")
        snapshot = torch.load(load_path)

        self.epoch = snapshot['epoch']
        self.model.load_state_dict(snapshot['model_state'])
        self.opt = snapshot['opt']

    def make_dataloader(self):
        if self.GNN:
            DL = pyg.loader.DataLoader
        else:
            DL = torch.utils.data.DataLoader

        if self.DDP:
            DS = torch.utils.data.distributed.DistributedSampler
            _shuffle, __shuffle = False, False
            _sampler, __sampler = DS(self._data), DS(self._data, shuffle=False)

            if self.data_ is not None:
                shuffle_ = False
                sampler_ = DS(self.data_, shuffle=False)
            else: # unused
                shuffle_ = False
                sampler_ = None
        else:
            _shuffle, __shuffle, shuffle_ = True, False, False
            _sampler, __sampler, sampler_ = None, None , None

        _args  = dict(shuffle= _shuffle, sampler= _sampler)
        __args = dict(shuffle=__shuffle, sampler=__sampler)
        args_  = dict(shuffle=shuffle_ , sampler=sampler_ )

        self._loader  = DL(self._data, batch_size=self._batch_size , collate_fn=self.collate_fn, **_args,)
        self._loader_ = DL(self._data, batch_size=self._batch_size_, collate_fn=self.collate_fn, **__args,)

        if self.data_ is not None:
            self.loader_ = DL(self.data_, batch_size=self.batch_size_, collate_fn=self.collate_fn, **args_)
        else:
            self.loader_ = None

        ###
        # Printing
        ###
        if self.verbose and self.LOCAL_RANK == 0:
            print(f"Number of training samples: {len(self._data)}")
            if self.data_ is not None:
                print(f"Number of test samples: {len(self.data_)}")
            else:
                print(f"No test data provided")

            if self.GNN:
                for batch in self._loader:
                    print(batch)
                    break
                if self.data_ is not None:
                    for batch in self.loader_:
                        print(batch)
                        break
            else:
                for (x, y) in self._loader:
                    print(f"Shape of x: {x.shape} {x.dtype}")
                    print(f"Shape of u: {y.shape} {y.dtype}")
                    break
        return

    def train(self):
        self.make_dataloader()
        self.statistics()

        while self.epoch < self.nepochs:
            self.epoch += 1

            # self.trigger_callbacks("epoch_start")
            self.train_epoch()
            # self.trigger_callbacks("epoch_end")

            if (self.epoch % self.stats_every) == 0:
                self.statistics()

        return

    def train_epoch(self):
        self.model.train()

        if self.DDP:
            self._loader.sampler.set_epoch(self.epoch)

        print_batch = self.verbose and (self.LOCAL_RANK == 0) and (len(self._loader) > 1) and self.print_batch

        if print_batch:
            batch_iterator = tqdm(
                self._loader,
                bar_format='{desc}{n_fmt}/{total_fmt} {bar}[{rate_fmt}]',
            )
        else:
            batch_iterator = self._loader

        for batch in batch_iterator:
            self.opt.zero_grad()
            self.trigger_callbacks("batch_start")
            loss = self.batch_loss(batch)
            loss.backward()
            self.trigger_callbacks("batch_post_grad")
            self.opt.step()
            self.schedule.step()
            self.trigger_callbacks("batch_end")
            self.opt.zero_grad()

            if print_batch:
                batch_iterator.set_description(
                    f"[Epoch {self.epoch} / {self.nepochs}] " +
                    f"LR {self.schedule.get_last_lr()[0]:.2e} " +
                    f"LOSS {loss.item():.8e}"
                )
        #
        return

    def batch_loss(self, batch):
        if self.GNN:
            batch = batch.to(self.device)
            yh = self.model(batch)
            loss = self.lossfun(yh, batch.y)
        else:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)

            yh = self.model(x)
            loss = self.lossfun(yh, y)
        #
        return loss

    def batch_size(self, batch):
        if self.GNN:
            return batch.y.size(0)
        else:
            return batch[1].size(0)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        N, L = 0, 0.0
        for batch in loader:
            n = self.batch_size(batch)
            l = self.batch_loss(batch).item()
            N += n
            L += l * n

        if self.DDP:
            L = torch.tensor(L, device=self.device)
            N = torch.tensor(N, device=self.device)
            dist.all_reduce(L, dist.ReduceOp.SUM)
            dist.all_reduce(N, dist.ReduceOp.SUM)
            loss = L.item() / N.item()
        else:
            loss = L / N

        return loss, None

    def statistics(self):
        _loss, _stats = self.evaluate(self._loader_)

        if self.loader_ is not None:
            loss_, stats_ = self.evaluate(self.loader_)
        else:
            loss_, stats_ = _loss, _stats

        # printing
        if self.print_epoch and self.verbose and (self.LOCAL_RANK == 0):
            msg = f"[Epoch {self.epoch} / {self.nepochs}] "
            if self.loader_ is not None:
                msg += f"TRAIN LOSS: {_loss:.6e} | TEST LOSS: {loss_:.6e}"
            else:
                msg += f"LOSS: {_loss:.6e}"
            if _stats is not None:
                if self.loader_ is not None:
                    msg += f"TRAIN STATS: {_stats} | TEST STATS: {stats_}"
                else:
                    msg += f"STATS: {_stats}"
            print(msg)

        return (_loss, _stats), (loss_, stats_)
#
