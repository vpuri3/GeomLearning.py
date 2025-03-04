#
import os
import json

import torch
import shutil
from tqdm import tqdm

import mlutils

__all__ = [
    'Callback',
]

#======================================================================#
class Callback:
    def __init__(self, case_dir: str, save_every=None):
        self.case_dir = case_dir
        self.save_every = save_every
        self.final = False

    def get_ckpt_dir(self, trainer: mlutils.Trainer):
        if self.final:
            ckpt_dir = os.path.join(self.case_dir, f'final')
        else:
            nsave = trainer.epoch // self.save_every
            ckpt_dir = os.path.join(self.case_dir, f'ckpt{str(nsave).zfill(2)}')

        if os.path.exists(ckpt_dir) and trainer.GLOBAL_RANK == 0:
            print(f"Removing {ckpt_dir}")
            shutil.rmtree(ckpt_dir)

        return ckpt_dir

    def load(self, trainer: mlutils.Trainer):
        ckpt_dirs = [dir for dir in os.listdir(self.case_dir) if dir.startswith('ckpt')]
        if len(ckpt_dirs) == 0:
            if trainer.GLOBAL_RANK == 0:
                print(f'No checkpoint found in {self.case_dir}. starting from scrach.')
            return
        load_dir = sorted(ckpt_dirs)[-1]
        model_file = os.path.join(self.case_dir, load_dir, 'model.pt')

        trainer.load(model_file)

        return

    @torch.no_grad()
    def __call__(self, trainer: mlutils.Trainer, final: bool=False):

        #------------------------#
        self.final = final
        if not self.final:
            if self.save_every is None:
                self.save_every = trainer.stats_every
            if trainer.epoch == 0:
                # return
                pass
            if (trainer.epoch % self.save_every) != 0:
                return
        #------------------------#

        # save model
        ckpt_dir = self.get_ckpt_dir(trainer)
        if trainer.GLOBAL_RANK == 0:
            print(f"saving checkpoint to {ckpt_dir}")
            os.makedirs(ckpt_dir, exist_ok=True)
            trainer.save(os.path.join(ckpt_dir, 'model.pt'))

        # save stats
        if trainer.GLOBAL_RANK == 0:
            print(f"Saving stats to {ckpt_dir}/stats.json")
            with open(os.path.join(ckpt_dir, 'stats.json'), 'w') as f:
                json.dump(trainer.stat_vals, f)

        # evaluate model
        self.evaluate(trainer, ckpt_dir)

        # revert self.final
        self.final = False

        return

    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str):
        batch = trainer._data[:]
        x = batch[0].to(trainer.device)
        y = batch[1].to(trainer.device)
        yh, slice_weights, temperature, attn_weights = trainer.model(x, return_stats=True)
        mse = (yh - y).pow(2).mean()
        
        # Compute sparsity of slice weights
        slice_sparsity = [ (w.abs() < 1e-2).float().mean().item() for w in slice_weights ]
        attn_sparsity = [ (w.abs() < 1e-2).float().mean().item() for w in attn_weights ]                

        # Compute mean and std of temperature
        temp_means = [t.mean().item() for t in temperature]
        temp_stds = [t.std().item() for t in temperature]

        print()
        print(f"Temperature means per layer: {[round(t, 4) for t in temp_means]}")
        print(f"Temperature stds per layer: {[round(t, 4) for t in temp_stds]}")
        print(f"Slice sparsity per layer: {[round(s, 4) for s in slice_sparsity]}. Mean: {sum(slice_sparsity) / len(slice_sparsity):.4f}")
        print(f"Attn sparsity per layer: {[round(s, 4) for s in attn_sparsity]}. Mean: {sum(attn_sparsity) / len(attn_sparsity):.4f}")
        print(f"Train MSE: {mse.item():.4f}")
        print()

        return

#======================================================================#
#