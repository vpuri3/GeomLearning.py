#
import os
import torch
import torch_geometric as pyg
import shutil
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlutils

from .time_march import march_case
from .visualize import (visualize_pyv, visualize_timeseries_pyv)

__all__ = [
    'FinaltimeCallback',
    'TimeseriesCallback',
]

#======================================================================#
class Callback:
    def __init__(self, case_dir: str, mesh: bool, save_every=None, num_eval_cases=None):
        self.case_dir = case_dir
        self.mesh = mesh
        self.save_every = save_every
        self.num_eval_cases = num_eval_cases if num_eval_cases is not None else 20
        self.final = False

    def get_ckpt_dir(self, trainer: mlutils.Trainer):
        if self.final:
            ckpt_dir = os.path.join(self.case_dir, f'final')
        else:
            ckpt_dirs = [dir for dir in os.listdir(self.case_dir) if dir.startswith('ckpt')]
            nsave = len(ckpt_dirs) + 1
            ckpt_dir = os.path.join(self.case_dir, f'ckpt{str(nsave).zfill(2)}')

        if os.path.exists(ckpt_dir):
            print(f"Removing {ckpt_dir}")
            shutil.rmtree(ckpt_dir)

        return ckpt_dir

    def load(self, trainer: mlutils.Trainer):
        ckpt_dirs = [dir for dir in os.listdir(self.case_dir) if dir.startswith('ckpt')]
        if len(ckpt_dirs) == 0:
            if trainer.LOCAL_RANK == 0:
                print(f'No checkpoint found in {self.case_dir}. starting from scrach.')
            return
        load_dir = sorted(ckpt_dirs)[-1]
        model_file = os.path.join(self.case_dir, load_dir, 'model.pt')

        trainer.load(model_file)

        return

    def get_dataset_transform(self, dataset):
        if dataset is None:
            return None
        elif isinstance(dataset, torch.utils.data.Subset):
            return self.get_dataset_transform(dataset.dataset)
        elif isinstance(dataset, pyg.data.Dataset):
            return dataset.transform

    def modify_dataset_transform(self, trainer: mlutils.Trainer, val: bool):
        """
        modify transform to return mesh, original fields
        """
        for dataset in [trainer._data, trainer.data_]:
            if dataset is None:
                continue

            transform = self.get_dataset_transform(dataset)
            transform.mesh = True if val else self.mesh
            transform.orig = val
            transform.elems = val
            transform.metadata = val

        return

    @torch.no_grad()
    def __call__(self, trainer: mlutils.Trainer, final: bool=False):

        #------------------------#
        if trainer.LOCAL_RANK != 0:
            return

        #------------------------#
        self.final = final
        if not self.final:
            if self.save_every is None:
                self.save_every = trainer.stats_every
            if trainer.epoch == 0:
                return
            if (trainer.epoch % self.save_every) != 0:
                return
        #------------------------#

        # save model
        ckpt_dir = self.get_ckpt_dir(trainer)
        print(f"saving checkpoint to {ckpt_dir}")
        os.makedirs(ckpt_dir, exist_ok=True)
        trainer.save(os.path.join(ckpt_dir, 'model.pt'))

        # update data transform
        self.modify_dataset_transform(trainer, True)

        # evaluate model
        self.evaluate(trainer, ckpt_dir)

        # revert data transform
        self.modify_dataset_transform(trainer, False)

        # revert self.final
        self.final = False

        return

#======================================================================#
class FinaltimeCallback(Callback):
    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str):

        device = trainer.device
        model  = trainer.model.module if trainer.DDP else trainer.model

        for (dataset, transform, split) in zip(
            [trainer._data, trainer.data_],
            [self.get_dataset_transform(trainer._data), self.get_dataset_transform(trainer.data_)],
            ['train', 'test'],
        ):
            if dataset is None:
                print(f"No {split} dataset.")
                continue
            else:
                print(f"Evaluating {split} dataset.")

            stats_file = os.path.join(ckpt_dir, f'{split}_stats.txt')

            case_nums = []
            case_names = []
            l2s = []
            r2s = []
            
            for icase in tqdm(range(len(dataset))):
                data = dataset[icase].to(device)
                data.yh = model(data)
                data.e = data.y - data.yh
                data.yp = data.yh * transform.scale.to(device)

                case_nums.append(str(icase).zfill(4))
                case_names.append(data.metadata['case_name'])
                l2s.append(torch.nn.MSELoss()(data.yh, data.y).item())
                r2s.append(mlutils.r2(data.yh, data.y))

                if self.final and (icase < self.num_eval_cases):
                    name = f'{os.path.basename(self.case_dir)}-{split}{str(icase).zfill(4)}'
                    out_file = os.path.join(ckpt_dir, name + '.vtu')
                    visualize_pyv(data, out_file)

                del data

            df = pd.DataFrame({
                'case_num': case_nums,
                'case_name': case_names,
                'MSE': l2s,
                'R-Square': r2s
            })
            print(f"Saving {split} stats to {stats_file}")
            df.to_csv(stats_file, index=False)
            
        if self.final:
            r2_values = {'train': [], 'test': []}
            for split in ['train', 'test']:
                stats_file = os.path.join(ckpt_dir, f'{split}_stats.txt')
                df = pd.read_csv(stats_file)
                r2_values[split] = df['R-Square'].values
            
            plot_boxes(r2_values, filename=os.path.join(ckpt_dir, 'r2_boxplot.png'))

        return

#======================================================================#
class TimeseriesCallback(Callback):
    def __init__(
        self, case_dir: str, save_every=None, num_eval_cases=None,
        autoreg_start=1,
    ):
        super().__init__(case_dir, save_every ,num_eval_cases)
        self.autoreg_start = autoreg_start

    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str):

        device = trainer.device
        model  = trainer.model.module if trainer.DDP else trainer.model

        for (dataset, transform, split) in zip(
            [trainer._data, trainer.data_],
            [self.get_dataset_transform(trainer._data), self.get_dataset_transform(trainer.data_)],
            ['train', 'test'],
        ):
            if dataset is None:
                print(f"No {split} dataset.")
                continue
            else:
                print(f"Evaluating {split} dataset.")

            C = min(self.num_eval_cases, len(dataset.case_files))
            cases = [dataset[dataset.case_range(c)] for c in range(C)]

            for icase in tqdm(range(len(cases))):
                case = cases[icase]
                case_data = cases[icase]
                ii = str(icase).zfill(4)
    
                for (autoreg, ext) in zip([True, False], ['AR', 'NR']):
                    eval_data, l2s, r2s = march_case(
                        model, case_data, transform,
                        autoreg=autoreg, device=device, K=self.autoreg_start,
                    )
    
                    name = f'{os.path.basename(self.case_dir)}-{split}{ii}-{ext}'
                    case_dir = os.path.join(ckpt_dir, name)
                    if self.final:
                        visualize_timeseries_pyv(eval_data, case_dir, merge=True, name=name)
    
                    out_file = os.path.join(ckpt_dir, f'{name}_stats.txt')
                    with open(out_file, 'w') as f:
                        f.write('Step\tMSE\tR-Square\n')
                        for (k, (l2, r2)) in enumerate(zip(l2s, r2s)):
                            f.write(f'{k}\t{l2s[k]:.8e}\t{r2s[k]}\n')

                    del eval_data 
                del case_data

        return

#======================================================================#
def plot_boxes(
    vals,
    titles=dict(train="Training", test="Testing", od="Out-of-Dist."),
    lims=[-1, 1],
    filename=None,
    dpi=175,
):
    n = len(vals)
    plt.figure(figsize=(2*n, 3.4), dpi=dpi)

    vals_list = []
    ticklocs = []
    ticklabels = []

    for i, key in enumerate(vals):
        vals_list.append(vals[key])
        ticklocs.append(i)
        ticklabels.append(f"{titles[key]}, N={len(vals[key])}")

    plt.boxplot(vals_list, positions=ticklocs)
    plt.xticks(ticklocs, ticklabels)
    plt.ylabel('R-Squared')
    plt.ylim(lims)
    plt.xlim(ticklocs[0]-0.5, ticklocs[-1]+0.5)
    plt.plot([ticklocs[0]-0.5, ticklocs[-1]+0.5], [0, 0],'k-', linewidth=0.5, zorder=-1)

    plt.savefig(filename, bbox_inches = "tight")

    return

#======================================================================#
