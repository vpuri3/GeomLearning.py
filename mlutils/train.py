#
# 3rd party
import torch
from torch import nn
from torch.utils.data import DataLoader

# builtin
import math

# local
from .utils import num_parameters

class Trainer:
    def __init__(
        self, 
        model,
        _data,
        data_,

        device="cpu",

        _batch_size=None,
        batch_size_=None,
        __batch_size=None,

        lr=None,
        Opt=None,
        weight_decay=None,

        lossfun=None,
        nepochs=None,

        statsfun=None,

        wandb=False,
    ):

        # DATA
        if _batch_size is None:
            _batch_size = 32
        if __batch_size is None:
            __batch_size = len(_data)
        if batch_size_ is None:
            batch_size_ = len(data_)

        _loader  = DataLoader(_data, batch_size=_batch_size)
        __loader = DataLoader(_data, batch_size=__batch_size, shuffle=False)
        loader_  = DataLoader(data_, batch_size=batch_size_ , shuffle=False)

        for (x, y) in _loader:
            print(f"Shape of x: {x.shape}")
            print(f"Shape of u: {y.shape} {y.dtype}")
            break

        # MODEL
        print(model)
        print(f"number of parameters: {num_parameters(model)}")
        model.to(device)

        # OPTIMIZER
        param = model.parameters()

        if lr is None:
            lr = 1e-3
        if Opt is None:
            Opt = torch.optim.Adam
        if Opt == torch.optim.Adam or Opt == torch.optim.AdamW:
            if weight_decay is None:
                if Opt == torch.optim.Adam:
                    weight_decay = 0.0
                else:
                    weight_decay = 0.01
            opt = torch.optim.Adam(param, lr=lr, weight_decay=weight_decay)

        if lossfun is None:
            lossfun = nn.MSELoss()
        if nepochs is None:
            nepochs = 100

        # ASSIGN TO SELF
        self.device = device

        self._loader = _loader
        self.__loader = __loader
        self.loader_ = loader_

        self.model = model

        self.opt = opt

        self.lossfun = lossfun
        self.nepochs = nepochs

        self.statsfun = statsfun

        self.wandb = wandb

        return

    def train(self):

        if self.wandb:
            wandb.config({
                **(wandb.config),

                "data_size" : len(_loader.dataset),
                "num_batches" : len(self._loader),
                "batch_size" : self._loader.batch_size,

                "num_parameters" : num_parameters(self.model),

                "learning_rate" : self.opt.lr,
                "weight_decay" : self.opt.weight_decay,

                "nepochs" : self.nepochs,
                "lossfun" : str(self.lossfun),
            })
        #

        # log initial metrics
        _loss, _stats = self.evaluate(self.__loader)
        loss_, stats_ = self.evaluate(self.loader_ )

        if self.wandb:
            wandb.log({
                "epoch" : 0,
                "train_loss" : _loss,
                "test_loss"  : loss_,
                # "train_stats" : _stats,
                # "test_stats"  : stats_,
            })
        #

        for epoch in range(1, self.nepochs + 1):
            print(f"-------------------------------")
            print(f"Epoch {epoch}")
            print(f"-------------------------------")

            self.train_epoch()
            _loss, _stats = self.evaluate(self._loader)
            loss_, stats_ = self.evaluate(self.loader_)

            if self.wandb:
                wandb.log({
                    "epoch" : 0,
                    "train_loss" : _loss,
                    "test_loss"  : loss_,
                })

                if not _stats:
                    wandb.log({
                        "train_stats" : _stats,
                        "test_stats"  : stats_,
                    })
            #
            print()
            print(f"\t TRAIN LOSS: {_loss:>10f} " + f"TEST LOSS: {loss_:>10f}")
            print()
            # if _stats:
            #     pass
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
            self.opt.zero_grad()

            if batch % printbatch == 0:
                print(
                    f"[{batch:>5d} / {nbatches:>5d}] \t" +
                    f" BATCH LOSS = {loss.item():>10f}"
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
#
