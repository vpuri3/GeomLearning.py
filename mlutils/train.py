#
# 3rd party
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# builtin
import math
import time # todo

# local
from mlutils.utils import num_parameters

__all__ = [
    'Trainer',
]

class Trainer:
    def __init__(
        self, 
        model,
        _data,
        data_,

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
    ):

        if device is None:
            device = "cpu"

        # DATA
        if _batch_size is None:
            _batch_size = 32
        if __batch_size is None:
            __batch_size = len(_data)
        if batch_size_ is None:
            batch_size_ = len(data_)

        # TODO: pin_memory=True, pin_memory_device=device
        # would then remove batch.to(device) calls in training loop

        _loader  = DataLoader(_data, batch_size=_batch_size)
        __loader = DataLoader(_data, batch_size=__batch_size, shuffle=False)
        loader_  = DataLoader(data_, batch_size=batch_size_ , shuffle=False)

        for (x, y) in _loader:
            print(f"Shape of x: {x.shape} {x.dtype}")
            print(f"Shape of u: {y.shape} {y.dtype}")
            break

        # MODEL
        print(model)
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
            schedule = optim.lr_scheduler.OneCycleLR(opt, 1e-1, epochs=nepochs, steps_per_epoch=len(_loader))
        elif Schedule is None:
            schedule = optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=1e10)
        else:
            raise NotImplementedError()

        config = {
            "device" : device,

            "data_size" : len(_loader.dataset),
            "num_batches" : len(_loader),
            "batch_size" : _loader.batch_size,

            "num_parameters" : num_parameters(model),

            "learning_rate" : lr,
            "weight_decay" : weight_decay,
            "optimizer" : str(opt),
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
        self.device = device

        self._loader  = _loader
        self.__loader = __loader
        self.loader_  = loader_

        self.model = model

        self.opt = opt
        self.schedule = schedule

        self.lossfun = lossfun
        self.nepochs = nepochs

        self.statsfun = statsfun

        self.wandb = wandb
        self.config = config

        return

    def train(self):

        self.print_train_banner(0, 0)
        self.callback()

        for epoch in range(1, self.nepochs + 1):
            self.print_train_banner(epoch, self.nepochs)
            self.train_epoch()
            self.callback()
        #

        return

    def train_epoch(self):
        self.model.train()
        nbatches = len(self._loader)
        printbatch = math.floor(nbatches / 10)

        for (batch, (x, u)) in enumerate(self._loader):
            x, u = x.to(self.device), u.to(self.device)

            uh = self.model(x)
            loss = self.lossfun(uh, u)

            loss.backward()
            self.opt.step()
            self.schedule.step()
            self.opt.zero_grad()

            if batch % printbatch == 0:
                print(
                    f"[{batch:>5d} / {nbatches:>5d}]\t" +
                    f"LR: {self.opt.param_groups[0]['lr']:>.3e}\t" +
                    f" BATCH LOSS = {loss.item():>.8e}"
                )
            #
        #

        return

    def evaluate(self, loader):
        self.model.eval()

        stats = None
        avg_loss = 0

        # if statsfun:
        #     pass

        with torch.no_grad():
            for (x, u) in loader:
                x, u = x.to(self.device), u.to(self.device)
                uh = self.model(x)
                loss = self.lossfun(uh, u)
                avg_loss += loss.item()

                # if statsfun:
                #     pass
                #
            #
        #

        nbatches = len(loader)
        loss = avg_loss / nbatches

        return loss, stats

    def callback(self):
        _loss, _stats = self.evaluate(self._loader)
        loss_, stats_ = self.evaluate(self.loader_)

        print()
        print(f"\t TRAIN LOSS: {_loss:>.8e}, STATS: {_stats}")
        print(f"\t TEST  LOSS: {loss_:>.8e}, STATS: {stats_}")
        print()

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

        return
#
