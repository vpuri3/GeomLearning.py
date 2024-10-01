#
# 3rd party
import torch
from torch import nn, optim
import torch_geometric as pyg

from tqdm import tqdm

# builtin
import math
import time
import collections

# local
from mlutils.utils import num_parameters, select_device

__all__ = [
    'Trainer',
]

class Trainer:
    def __init__(
        self, 
        model,
        _data,
        data_=None,

        gnn=False,
        device=None,

        _batch_size=None,
        batch_size_=None,
        __batch_size=None,

        lr=None,
        weight_decay=None,

        Opt=None,
        Schedule=None,

        lossfun=None,
        nepochs=None,

        statsfun=None,

        wandb=False,
        verbose=True,
    ):

        # TODO: early stopping
        device = select_device(device, verbose=verbose)

        # DATA
        if _batch_size is None:
            _batch_size = 32
        if __batch_size is None:
            __batch_size = len(_data)
        if data_ is not None:
            if batch_size_ is None:
                batch_size_ = len(data_)

        # MODEL
        if verbose:
            # print(model)
            print(f"number of parameters: {num_parameters(model)}")
            print(f"Moving model to: {device}")
        model.to(device)

        # OPTIMIZER
        param = model.parameters()

        if lr is None:
            lr = 1e-3
        if weight_decay is None:
            weight_decay = 5e-4

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
            "gnn" : gnn,
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

        if wandb:
            config = {**(wandb.config), **config}
            wandb.config(config)

        print(f"Trainer config:")
        for (k, v) in config.items():
            print(f"{k} : {v}")

        # ASSIGN TO SELF

        # MISC
        self.gnn = gnn
        self.device = device
        self.verbose = verbose

        # DATA
        self._data = _data
        self.data_ = data_
        self._batch_size = _batch_size
        self.batch_size_ = batch_size_
        self.__batch_size = __batch_size

        # MODEL
        self.model = model

        # OPT
        self.opt = opt
        self.schedule = schedule

        self.lossfun = lossfun
        self.nepochs = nepochs
        self.statsfun = statsfun

        self.wandb = wandb
        self.config = config

        # callback
        self.callbacks = collections.defaultdict(list)

        # iteration
        self.epoch = 0
        self.start_time = 0.0
        self.epoch_time = 0.0
        self.epoch_dt   = 0.0

    # https://github.com/karpathy/minGPT/
    def add_callback(self, event: str, callback):
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        self.callbacks[event] = [callback]

    def trigger_callback(self, event: str):
        for callback in self.callbacks[event]: # self.callbacks.get(event, []):
            callback(self)

    def save(self, save_path: str):
        data = {
            "epoch": self.epoch,
            "model": self.model.state_dict(), # move to cpu first?
            # "opt": self.opt
        }
        torch.save(data, save_path)
        return

    def load(self, load_path: str):
        print(f"Loading {load_path}")
        data = torch.load(load_path)

        self.epoch = data["epoch"]
        self.model.load_state_dict(data["model"])
        self.model.to(self.device)

    def make_dataloader(self):
        # TODO: loader pin_memory=True, pin_memory_device=device
        # would then remove batch.to(device) calls in training loop
        # TODO: loader sampler (replacement = True)

        if self.gnn:
            DL = pyg.loader.DataLoader
        else:
            DL = torch.utils.data.DataLoader

        self._loader  = DL(self._data, batch_size=self._batch_size, shuffle=True)
        self.__loader = DL(self._data, batch_size=self.__batch_size, shuffle=False)
        if self.data_ is not None:
            self.loader_ = DL(self.data_, batch_size=self.batch_size_ , shuffle=False)
        else:
            self.loader_ = None

        if self.verbose:
            print(f"Number of training samples: {len(self._data)}")
            if self.data_ is not None:
                print(f"Number of test samples: {len(self.data_)}")
            else:
                print(f"No test data provided")

        if self.verbose:
            if self.gnn:
                for batch in self._loader:
                    print(batch)
                    break
            else:
                for (x, y) in self._loader:
                    print(f"Shape of x: {x.shape} {x.dtype}")
                    print(f"Shape of u: {y.shape} {y.dtype}")
                    break

        # _loader = DL(
        #     self.train_dataset,
        #     sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
        #     shuffle=False,
        #     pin_memory=True,
        #     batch_size=config.batch_size,
        #     num_workers=config.num_workers,
        # )

        return

    def train(self):
        # move to device here?
        self.make_dataloader()

        self.start_time = time.time()
        self.print_train_banner(self.epoch, self.nepochs)
        self.trigger_callback("epoch_end")
        self.statistics()

        while self.epoch < self.nepochs:
            self.epoch += 1
            self.print_train_banner(self.epoch, self.nepochs)

            self.epoch_time = time.time() - self.start_time
            self.train_epoch()
            self.epoch_dt = time.time() - self.epoch_time - self.start_time

            self.trigger_callback("epoch_end")
            self.statistics()
        #

        return

    def train_epoch(self):
        self.model.train()

        batch_iterator = tqdm(
            self._loader,
            bar_format='{n_fmt}/{total_fmt} {desc}{bar}[{rate_fmt}]',
        )

        for batch in batch_iterator:
            loss = self.batch_loss(batch)
            loss.backward()
            self.opt.step()
            self.schedule.step()
            self.opt.zero_grad()

            batch_iterator.set_description(
                f"LR: {self.schedule.get_last_lr()[0]:.2e} " +
                f"LOSS: {loss.item():.8e}"
            )
            self.trigger_callback("batch_end")
        #
        return

    def batch_loss(self, batch):
        if self.gnn:
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

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        stats = None
        avg_loss = 0

        # if statsfun:
        #     pass

        for batch in loader:
            loss = self.batch_loss(batch)
            avg_loss += loss.item()

            # if statsfun:
            #     pass
        #

        nbatches = len(loader)
        loss = avg_loss / nbatches

        return loss, stats

    def statistics(self):
        _loss, _stats = self.evaluate(self._loader)

        if self.loader_ is not None:
            loss_, stats_ = self.evaluate(self.loader_)
        else:
            loss_, stats_ = _loss, _stats

        print()
        print(f"\t TRAIN LOSS: {_loss:>.8e}, STATS: {_stats}")
        print(f"\t TEST  LOSS: {loss_:>.8e}, STATS: {stats_}")
        print()

        print(f"EPOCH START TIME: {self.epoch_time:.4f}s, DT: {self.epoch_dt:.4f}s")

        if self.wandb:
            wandb.log({
                "epoch" : 0,
                "train_loss" : _loss,
                "test_loss"  : loss_,
                "train_stats" : _stats,
                "test_stats"  : stats_,
            })
        #

        return (_loss, _stats), (loss_, stats_)

    @staticmethod
    def print_train_banner(epoch, nepochs):
        print(f"-------------------------------")
        print(f"Epoch {epoch} / {nepochs}")
        print(f"-------------------------------")

#
