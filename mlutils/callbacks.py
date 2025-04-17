#
import gc
import os
import json
import shutil
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt

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
            ckpt_dir = os.path.join(self.case_dir, f'eval')
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
            with open(os.path.join(ckpt_dir, 'stats.json'), 'w') as f:
                json.dump(trainer.stat_vals, f)

        # save loss plot
        if trainer.GLOBAL_RANK == 0:
            plt.figure(figsize=(8, 4), dpi=175)
            train_loss_per_batch = trainer.train_loss_per_batch
            if isinstance(train_loss_per_batch, list):
                train_loss_per_batch = torch.tensor(train_loss_per_batch)
            train_loss_per_batch[train_loss_per_batch < 1e-12] = torch.nan
            plt.plot(train_loss_per_batch, color='k', label='Train loss (per batch)', alpha=0.5)
            plt.plot(trainer.num_steps_fullbatch, trainer.train_loss_fullbatch, color='r', label='Train loss (full batch)', marker='o')
            plt.plot(trainer.num_steps_fullbatch, trainer.test_loss_fullbatch , color='b', label='Test loss (full batch)', marker='o')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.yscale('log')
            if trainer.stat_vals['train_loss'] is not None:
                plt.title(f'Train Loss (final): {trainer.stat_vals["train_loss"]:.2e}, Test Loss (final): {trainer.stat_vals["test_loss"]:.2e}')
            else:
                plt.title('Train Loss')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, 'losses.png'))
            plt.savefig(os.path.join(self.case_dir, 'losses.png'))
            plt.close()

        # modify dataset transform
        if hasattr(self, 'modify_dataset_transform'):
            self.modify_dataset_transform(trainer, True)

        # evaluate model
        self.evaluate(trainer, ckpt_dir)

        # revert dataset transform
        if hasattr(self, 'modify_dataset_transform'):
            self.modify_dataset_transform(trainer, False)

        # revert self.final
        self.final = False

        # clear cache
        if trainer.is_cuda:
            gc.collect()
            torch.cuda.empty_cache()

        return

    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str):
        return

#======================================================================#
#