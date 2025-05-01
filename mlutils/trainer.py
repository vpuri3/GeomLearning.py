#
# 3rd party
import torch
from torch import nn, optim
from torch import distributed as dist

from tqdm import tqdm

# builtin
import os
import collections

# local
from mlutils.schedule import DecayScheduler
from mlutils.utils import (
    num_parameters, select_device, is_torchrun, check_package_version_lteq,
)

__all__ = [
    'Trainer',
]

#======================================================================#

class Trainer:
    def __init__(
        self, 
        model,
        _data,
        data_=None,

        gnn_loader=False,
        device=None,

        collate_fn=None,
        _batch_size=None,  # bwd over _data
        batch_size_=None,  # fwd over data_
        _batch_size_=None, # fwd over _data

        lr=None,
        weight_decay=None,
        clip_grad_norm=None,

        make_optimizer=None, # (model, lr, weight_decay) -> optimizer
        Schedule=None,
        
        # OneCycleLR schedule
        one_cycle_pct_start=0.3,        # % of cycle spent increasing LR. Default: 0.3
        one_cycle_div_factor=25,        # initial_lr = max_lr/div_factor. Default: 25
        one_cycle_final_div_factor=1e4, # min_lr = initial_lr/final_div_factor Default: 1e4
        one_cycle_three_phase=False,    # first two phases will be symmetrical about pct_start third phase: initial_lr -> initial_lr/final_div_factor

        noise_schedule='linear',
        noise_init=0.1,
        noise_min=0.0,
        
        lossfun=None,
        batch_lossfun=None,
        epochs=None,

        statsfun=None,
        verbose=True,
        print_config=False,
        print_batch=True,
        print_epoch=True,
        stats_every=1, # stats every k epochs
    ):
        
        ###
        # PRINTING
        ###

        self.verbose = verbose
        self.print_config = print_config
        self.print_batch = print_batch
        self.print_epoch = print_epoch
        self.stats_every = stats_every if stats_every != 0 else 1

        ###
        # DEVICE
        ###

        self.DISTRIBUTED = is_torchrun()
        self.GLOBAL_RANK = int(os.environ['RANK']) if self.DISTRIBUTED else 0
        self.LOCAL_RANK = int(os.environ['LOCAL_RANK']) if self.DISTRIBUTED else 0
        self.WORLD_SIZE = int(os.environ['WORLD_SIZE']) if self.DISTRIBUTED else 1

        if self.DISTRIBUTED:
            assert dist.is_initialized()
            self.DDP = dist.get_world_size() > 1
            self.device = torch.device(self.LOCAL_RANK)
        else:
            self.DDP = False
            self.device = select_device(device, verbose=True)
            
        self.is_cuda = self.device not in ['cpu', torch.device('cpu')]

        ###
        # DATA
        ###

        if _data is None:
            raise ValueError('_data passed to Trainer cannot be None.')

        self._data = _data
        self.data_ = data_

        self._batch_size = 32 if _batch_size is None else _batch_size
        self._batch_size_ = len(_data) if _batch_size_ is None else _batch_size_
        self.batch_size_ = batch_size_

        if (data_ is not None) and (batch_size_ is None):
            self.batch_size_ = len(data_)

        self.collate_fn = collate_fn
        self.gnn_loader = gnn_loader

        ###
        # MODEL
        ###

        if self.verbose and (self.GLOBAL_RANK == 0):
            print(f"Moving model with {num_parameters(model)} parameters to device {device}")

        self.model = model.to(device)

        if self.DDP:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[device])

        ###
        # OPTIMIZER
        ###

        if lr is None:
            lr = 1e-3
        if weight_decay is None:
            weight_decay = 0.0
        if make_optimizer is None:
            self.opt = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.opt = make_optimizer(self.model, lr, weight_decay)

        self.clip_grad_norm = clip_grad_norm

        ###
        # LOSS CALCULATION
        ###

        self.lossfun = nn.MSELoss() if lossfun is None else lossfun
        self.batch_lossfun = batch_lossfun

        ###
        # iteration
        ###

        self.epoch = 0
        self.epochs = 100 if epochs is None else epochs

        bsize = self._batch_size * dist.get_world_size() if self.DISTRIBUTED else self._batch_size
        self.steps_per_epoch = len(_data) // bsize + 1
        self.total_steps = self.steps_per_epoch * self.epochs

        if Schedule == "OneCycleLR":
            self.schedule = optim.lr_scheduler.OneCycleLR(
                self.opt,
                max_lr=lr,
                epochs=epochs,
                steps_per_epoch=self.steps_per_epoch,
                pct_start=one_cycle_pct_start,
                div_factor=one_cycle_div_factor,
                final_div_factor=one_cycle_final_div_factor,
                three_phase=one_cycle_three_phase,
            )
            self.update_schedule_every_epoch = False
        elif Schedule == "CosineAnnealingLR":
            self.schedule = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.epochs, eta_min=0.)
            self.update_schedule_every_epoch = True
        elif Schedule is None:
            self.schedule = optim.lr_scheduler.ConstantLR(self.opt, factor=1.0, total_iters=1e10)
            self.update_schedule_every_epoch = True
        else:
            raise NotImplementedError()
        
        self.noise_schedule = DecayScheduler(
            total_steps=self.total_steps,
            decay_type=noise_schedule,
            init_val=noise_init,
            min_val=noise_min,
        )
        
        ###
        # STATISTICS
        ###

        self.statsfun = statsfun
        self.stat_vals = {
            "train_loss" : None,
            "test_loss" : None,
            "train_stats" : None,
            "test_stats" : None,
        }
        self.train_loss_per_batch = []
        self.num_steps_fullbatch  = []
        self.train_loss_fullbatch = []
        self.test_loss_fullbatch  = []
        
        ###
        # Callbacks
        ###

        self.callbacks = collections.defaultdict(list)

    #------------------------#
    # CALLBACKS
    #------------------------#

    # https://github.com/karpathy/minGPT/
    def add_callback(self, event: str, callback):
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        self.callbacks[event] = [callback]

    def trigger_callbacks(self, event: str):
        for callback in self.callbacks[event]:
            callback(self)

    #------------------------#
    # SAVE / LOAD
    #------------------------#

    def save(self, save_path: str): # call only if device==0
        if self.GLOBAL_RANK != 0:
            return

        snapshot = dict()
        snapshot['epoch'] = self.epoch
        if self.DDP:
            snapshot['model_state'] = self.model.module.state_dict()
        else:
            snapshot['model_state'] = self.model.state_dict()
        snapshot['opt_state'] = self.opt.state_dict()
        snapshot['schedule_state'] = None if (self.schedule is None) else self.schedule.state_dict()
        snapshot['train_loss_per_batch'] = self.train_loss_per_batch

        torch.save(snapshot, save_path)

        return

    def load(self, load_path: str):
        if self.GLOBAL_RANK == 0:
            print(f"Loading checkpoint {load_path}")

        if check_package_version_lteq('torch', '2.4'):
            snapshot = torch.load(load_path, map_location=self.device)
        else:
            snapshot = torch.load(load_path, weights_only=False, map_location=self.device)

        self.epoch = snapshot['epoch']
        if self.DDP:
            self.model.module.load_state_dict(snapshot['model_state'])
        else:
            self.model.load_state_dict(snapshot['model_state'])
        self.opt.load_state_dict(snapshot['opt_state'])
        self.schedule.load_state_dict(snapshot['schedule_state'])
        self.train_loss_per_batch = snapshot['train_loss_per_batch']
        
        # noise schedul
        self.noise_schedule.set_current_step(self.epoch * self.steps_per_epoch)

    #------------------------#
    # DATALOADER
    #------------------------#

    def make_dataloader(self):
        if self.gnn_loader:
            import torch_geometric as pyg
            DL = pyg.loader.DataLoader
        else:
            DL = torch.utils.data.DataLoader

        if self.DDP:
            DS = torch.utils.data.distributed.DistributedSampler
            _shuffle, _shuffle_ = False, False
            _sampler, _sampler_ = DS(self._data), DS(self._data, shuffle=False)

            if self.data_ is not None:
                shuffle_ = False
                sampler_ = DS(self.data_, shuffle=False)
            else:
                shuffle_ = False
                sampler_ = None
        else:
            _shuffle, _shuffle_, shuffle_ = True, False, False
            _sampler, _sampler_, sampler_ = None, None , None

        _args  = dict(shuffle=_shuffle , sampler=_sampler )
        _args_ = dict(shuffle=_shuffle_, sampler=_sampler_)
        args_  = dict(shuffle=shuffle_ , sampler=sampler_ )

        self._loader  = DL(self._data, batch_size=self._batch_size , collate_fn=self.collate_fn, **_args )
        self._loader_ = DL(self._data, batch_size=self._batch_size_, collate_fn=self.collate_fn, **_args_)

        if self.data_ is not None:
            self.loader_ = DL(self.data_, batch_size=self.batch_size_, collate_fn=self.collate_fn, **args_)
        else:
            self.loader_ = None

        ###
        # Printing
        ###

        if self.verbose and self.print_config and self.GLOBAL_RANK == 0:
            print(f"Number of training samples: {len(self._data)}")
            if self.data_ is not None:
                print(f"Number of test samples: {len(self.data_)}")
            else:
                print(f"No test data provided")

            if self.gnn_loader:
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

    #------------------------#
    # TRAINING
    #------------------------#

    def train(self):
        self.make_dataloader()

        self.trigger_callbacks("epoch_start")
        self.statistics()
        self.trigger_callbacks("epoch_end")
        
        while self.epoch < self.epochs:
            self.epoch += 1

            self.trigger_callbacks("epoch_start")
            self.train_epoch()
            if (self.epoch % self.stats_every) == 0:
                self.statistics()
            self.trigger_callbacks("epoch_end")
            if self.update_schedule_every_epoch:
                self.schedule.step()

        return

    def train_epoch(self):
        if self.DDP:
            self._loader.sampler.set_epoch(self.epoch)

        print_batch = self.verbose and (self.GLOBAL_RANK == 0) and self.print_batch # and (len(self._loader) > 1)

        if print_batch:
            batch_iterator = tqdm(
                self._loader,
                bar_format='{desc}{n_fmt}/{total_fmt} {bar}[{rate_fmt}]',
                ncols=80,
            )
        else:
            batch_iterator = self._loader

        for batch in batch_iterator:
            self.opt.zero_grad()
            self.trigger_callbacks("batch_start")

            self.noise_schedule.step()

            self.model.train()
            loss = self.batch_loss(batch)
            loss.backward()

            self.train_loss_per_batch.append(loss.item())

            self.trigger_callbacks("batch_post_grad")
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            self.opt.step()
            self.trigger_callbacks("batch_end")
            self.opt.zero_grad()
            
            # update schedule every batch
            if not self.update_schedule_every_epoch:
                self.schedule.step()

            if print_batch:
                batch_iterator.set_description(
                    f"[Epoch {self.epoch} / {self.epochs}] " +
                    f"LR {self.schedule.get_last_lr()[0]:.2e} " +
                    f"LOSS {loss.item():.8e}"
                )

        return
    
    def to_device(self, x):
        return x.to(self.device) if isinstance(x, torch.Tensor) else x

    def batch_loss(self, batch):
        if self.batch_lossfun is not None:
            if isinstance(batch, tuple) or isinstance(batch, list):
                batch = [self.to_device(x) for x in batch]
            elif isinstance(batch, dict):
                batch = {k: self.to_device(v) for k, v in batch.items()}
            elif isinstance(batch, torch.Tensor):
                batch = self.to_device(batch)
            else:
                batch = batch.to(self.device)
            loss = self.batch_lossfun(self, self.model, batch)
        elif self.gnn_loader:
            batch = batch.to(self.device)
            yh = self.model(batch)
            loss = self.lossfun(yh, batch.y)
        else:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)

            yh = self.model(x)
            loss = self.lossfun(yh, y)

        return loss

    #------------------------#
    # STATISTICS
    #------------------------#

    def get_batch_size(self, batch, loader):
        bs = batch.num_graphs if self.gnn_loader else batch[0].size(0)
        return min(bs, loader.batch_size)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        N, L = 0, 0.0
        for batch in loader:
            n = self.get_batch_size(batch, loader)
            l = self.batch_loss(batch).item()
            N += n
            L += l * n
            
        if self.DDP:
            L = torch.tensor(L, device=self.device)
            N = torch.tensor(N, device=self.device)
            dist.all_reduce(L, dist.ReduceOp.SUM)
            dist.all_reduce(N, dist.ReduceOp.SUM)
            L, N = L.item(), N.item()

        if N == 0:
            loss = float('nan')
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
        if self.print_epoch and self.verbose and (self.GLOBAL_RANK == 0):
            msg = f"[Epoch {self.epoch} / {self.epochs}] "
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

        self.stat_vals = {
            "train_loss" : _loss,
            "test_loss" : loss_,
            "train_stats" : _stats,
            "test_stats" : stats_,
        }
        self.train_loss_fullbatch.append(_loss)
        self.test_loss_fullbatch.append(loss_)
        self.num_steps_fullbatch.append(len(self.train_loss_per_batch))
        
        return
#======================================================================#
#
