#
import os
import json

import torch
import torch_geometric as pyg

import gc
import shutil
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlutils
import bench

from bench.rollout import rollout
from am.callbacks import timeseries_statistics_plot
from am.callbacks import hstack_dataframes_across_ranks, vstack_dataframes_across_ranks

__all__ = [
    'TimeseriesCallback',
    'SparsityCallback',
    'SteadyStateCallback',
]

#======================================================================#
class TimeseriesCallback(mlutils.Callback):
    def __init__(
        self,
        case_dir: str, save_every=None,
        num_eval_cases=None, mesh=False, cells=False,
    ):
        super().__init__(case_dir, save_every=save_every)
        self.num_eval_cases = num_eval_cases
        self.mesh = mesh
    
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
            transform.cells = val
            transform.metadata = val
            
        return

    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str):
        
        if not self.final:
            if trainer.epoch / trainer.epochs < 0.5:
                return

        model  = trainer.model.module if trainer.DDP else trainer.model

        for (dataset, transform, split) in zip(
            [trainer._data, trainer.data_],
            [self.get_dataset_transform(trainer._data), self.get_dataset_transform(trainer.data_)],
            ['train', 'test'],
        ):
            if dataset is None:
                if trainer.GLOBAL_RANK == 0:
                    print(f"No {split} dataset.")
                continue
            
            split_dir = os.path.join(ckpt_dir, f'vis_{split}')

            if trainer.GLOBAL_RANK == 0:
                os.makedirs(split_dir, exist_ok=True)
            
            # distribute cases across ranks
            num_cases = dataset.num_cases
            cases_per_rank = num_cases // trainer.WORLD_SIZE 
            icase0 = trainer.GLOBAL_RANK * cases_per_rank
            icase1 = (trainer.GLOBAL_RANK + 1) * cases_per_rank if trainer.GLOBAL_RANK != trainer.WORLD_SIZE - 1 else num_cases
            
            case_nums = []

            l2_cases = []
            r2_cases = []

            if trainer.GLOBAL_RANK == 0:
                pbar = tqdm(total=num_cases, desc=f"Evaluating {split} dataset", ncols=80)
            
            for icase in range(icase0, icase1):
                case_idx = dataset.case_range(icase)
                case_data = dataset[case_idx]
                
                assert len(case_data) == dataset.num_steps, f"got {len(case_data)} steps, expected {dataset.num_steps} steps for case_idx = {case_idx}"

                eval_data, l2s, r2s = rollout(model, case_data, transform, init_step=dataset.init_step)

                # case_dir = os.path.join(split_dir, f"{split}{str(icase).zfill(3)}-{ext}")
                # file_name = f'{os.path.basename(self.case_dir)}-{split}{str(icase).zfill(4)}-{ext}'
                # if self.final and len(case_nums) < self.num_eval_cases:
                #     visualize_timeseries_pyv(eval_data, case_dir, merge=True, name=file_name)

                case_nums.append(icase)
                l2_cases.append(l2s)
                r2_cases.append(r2s)

                del eval_data 
                del case_data
                
                if trainer.GLOBAL_RANK == 0:
                    pbar.update(trainer.WORLD_SIZE)
                    
            if trainer.GLOBAL_RANK == 0:
                pbar.close()

            # Convert list of stats arrays into a DataFrame where each row represents
            # a time step and each column represents a case
            df_l2 = pd.DataFrame(l2_cases).transpose()
            df_r2 = pd.DataFrame(r2_cases).transpose()

            # Assign case numbers as column names
            df_l2.columns = case_nums
            df_r2.columns = case_nums

            # Assign step numbers as index
            df_l2.index.name = 'Step'
            df_r2.index.name = 'Step'

            # create dataframe for each autoreg
            df_l2 = hstack_dataframes_across_ranks(df_l2, trainer)
            df_r2 = hstack_dataframes_across_ranks(df_r2, trainer)
            
            if trainer.GLOBAL_RANK == 0:
                print(f"Saving {split} statistics to {ckpt_dir}")
                df_l2.to_csv(os.path.join(ckpt_dir, f'l2_stats_{split}.txt'), index=False)
                df_r2.to_csv(os.path.join(ckpt_dir, f'r2_stats_{split}.txt'), index=False)
            
        if trainer.DDP:
            torch.distributed.barrier()

        for split in ['train', 'test']:
            df_l2 = pd.read_csv(os.path.join(ckpt_dir, f'l2_stats_{split}.txt'))
            df_r2 = pd.read_csv(os.path.join(ckpt_dir, f'r2_stats_{split}.txt'))

            if trainer.GLOBAL_RANK == 0:
                print(f"Saving L2/R2 plots to {ckpt_dir}/r2_plot_{split}.png")
                timeseries_statistics_plot(df_r2, 'r2', 'mean', filename=os.path.join(ckpt_dir, f'r2_plot_{split}.png'))
                timeseries_statistics_plot(df_l2, 'l2', 'mean', filename=os.path.join(ckpt_dir, f'l2_plot_{split}.png'))

                timeseries_statistics_plot(df_r2, 'r2', 'mean', filename=os.path.join(ckpt_dir, '..', f'r2_plot_{split}.png'))
                timeseries_statistics_plot(df_l2, 'l2', 'mean', filename=os.path.join(ckpt_dir, '..', f'l2_plot_{split}.png'))
                
        return

#======================================================================#
class SteadyStateCallback(mlutils.Callback):
    def __init__(self, case_dir: str, y_normalizer):
        super().__init__(case_dir)
        self.y_normalizer = y_normalizer

    @torch.no_grad()
    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str):
        
        trainer.model.eval()
        
        y_normalizer = self.y_normalizer.to(trainer.device)
        
        # N, MSE = 0, 0.0

        # for batch in trainer._loader_:
        #     x = batch[0].to(trainer.device)
        #     y = batch[1].to(trainer.device)
        #     yh = trainer.model(x)
            
        #     n = trainer.get_batch_size(batch, trainer._loader_)
        #     N += n
        #     MSE += ((yh - y).pow(2).mean() * n).item()
                
        #     del x, y, yh

        # MSE = MSE / N

        # print(f"Train MSE: {MSE:.8e}")

        test_loss = bench.TestLoss()
        
        N, L = 0, 0.0
        rel_error = 0
        for batch in trainer.loader_:
            n = trainer.get_batch_size(batch, trainer.loader_)
            N += n
            x = batch[0].to(trainer.device)
            y = batch[1].to(trainer.device)

            yh = trainer.model(x)
            yh = y_normalizer.decode(yh)
            y  = y_normalizer.decode(y)
            rel_loss = test_loss.rel(yh,y)
            rel_error += rel_loss.item()
            print(f'{n}', f'{rel_loss.item():.8e}')

        rel_error = rel_error #/ N
        print(f'Relative Error (test): {rel_error:.8e}')

        return

#======================================================================#
class SparsityCallback(mlutils.Callback):
    # def __init__(self, case_dir: str, save_every=None):
    #     super().__init__(case_dir, save_every)

    @torch.no_grad()
    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str):
        
        trainer.model.eval()
        
        N, MSE = 0, 0.0
        slice_weights = [[] for _ in range(trainer.model.num_layers)]
        temperature = [[] for _ in range(trainer.model.num_layers)]
        bias = [[] for _ in range(trainer.model.num_layers)]
        attn_weights = [[] for _ in range(trainer.model.num_layers)]

        for batch in trainer._loader_:
            x = batch[0].to(trainer.device)
            y = batch[1].to(trainer.device)
            yh, swt, tmp, bs, att = trainer.model(x, return_stats=True)
            
            n = trainer.get_batch_size(batch, trainer._loader_)
            N += n
            MSE += ((yh - y).pow(2).mean() * n).item()
            for i in range(trainer.model.num_layers):
                slice_weights[i].append(swt[i].detach().cpu())
                temperature[i].append(tmp[i].detach().cpu())
                bias[i].append(bs[i].detach().cpu())
                attn_weights[i].append(att[i].detach().cpu())
                
            del x, y, yh, swt, tmp, att

        MSE = MSE / N
        slice_weights = [torch.cat(slice_weights[i], dim=0) for i in range(trainer.model.num_layers)]
        attn_weights = [torch.cat(attn_weights[i], dim=0) for i in range(trainer.model.num_layers)]
        temperature = [temperature[i][0] for _ in range(trainer.model.num_layers)]
        bias = [bias[i][0] for _ in range(trainer.model.num_layers)]

        # Attention utilization (number of non-zero attn weights)
        attn_utilization = [(w > 1e-2).sum(dim=-1).float().mean().item() / trainer.model.num_slices for w in attn_weights ]                
        
        print()
        print(f"Train MSE: {MSE:.8e}")
        # print(f"Noise level: {trainer.noise_schedule.get_current_val()}")
        print(f"K: {trainer.model.k_val}, Gamma: {trainer.model.gamma:.3e}")
        print(f"Attn utilization: {[round(s, 4) for s in attn_utilization]}. Mean: {sum(attn_utilization) / len(attn_utilization):.4f}")

        # print(f"Temperature stats:")
        # print(f"  Mean: {[round(t.mean().item(), 4) for t in temperature]}")
        # print(f"  Std : {[round(t.std().item(), 4) for t in temperature]}")
        # print(f"  Min : {[round(t.min().item(), 4) for t in temperature]}")
        # print(f"  Max : {[round(t.max().item(), 4) for t in temperature]}")

        # print(f"Bias stats:")
        # print(f"  Mean: {[round(b.mean().item(), 4) for b in bias]}")
        # print(f"  Std : {[round(b.std().item(), 4) for b in bias]}")
        # print(f"  Min : {[round(b.min().item(), 4) for b in bias]}")
        # print(f"  Max : {[round(b.max().item(), 4) for b in bias]}")

        # Slice sparsity: what proportion of slices are invoked per point
        slice_sparsity = [(w < 1e-2).sum(dim=-2).float().mean().item() * 100 / trainer.model.num_slices for w in slice_weights]
        
        # Slice utilization (how evenly are slices used)
        target_use = 1 / trainer.model.num_slices
        threshold = 0.5 * target_use
        slice_use = [w.mean(dim=-1) for w in slice_weights] # [B H M]
        underused = [(slice_use[i] < (target_use - threshold)).float().mean().item() * 100 for i in range(trainer.model.num_layers)]
        overused = [(slice_use[i] > (target_use + threshold)).float().mean().item() * 100 for i in range(trainer.model.num_layers)]

        # print(f"Slice utilization: How evenly are slices used?")
        # print(f"Want every slice to be used equally often, avoiding scenarios where")
        # print(f"some slices are always ignored (underutilized) or overly relied upon (overutilized).")
        # print(f"Sparsity: What proportion of slices are invoked per point?")

        print()
        print(f"Slice utilization stats:")
        print(f"  Mean: {[round(s.mean().item(), 4) for s in slice_use]} (Target: {target_use:.5f})")
        # print(f"  Std : {[round(s.std(dim=-1).mean().item(), 4) for s in slice_use]}")
        # print(f"  Min : {[round(s.min().item(), 4) for s in slice_use]}")
        print(f"  Max : {[round(s.max().item(), 4) for s in slice_use]}")
        # print(f"  % Underused (< {target_use - threshold:.5f}): {[round(u, 2) for u in underused]}. Mean: {sum(underused) / len(underused):.4f}")
        # print(f"  % Overused  (> {target_use + threshold:.5f}): {[round(o, 2) for o in overused]}. Mean: {sum(overused) / len(overused):.4f}")
        print(f"  % Sparsity : {[round(s, 2) for s in slice_sparsity]}. Mean: {sum(slice_sparsity) / len(slice_sparsity):.4f}")
        print()

        mean_slice_use = [w.mean(dim=[0,-1]) / target_use for w in slice_weights]
        slice_use_bias_corr = [torch.corrcoef(torch.stack([mean_slice_use[i].view(-1), bias[i].view(-1)])) for i in range(trainer.model.num_layers) ]

        # print(f"Slice use/ bias correlation: {[round(c[0,1].item(), 4) for c in slice_use_bias_corr]}")
        # print()

        fig, axes = plt.subplots(trainer.model.num_layers, 2, figsize=(10, 3*trainer.model.num_layers))
        fig.suptitle('Slice Utilization, Bias')
        
        for i in range(trainer.model.num_layers):

            corr = slice_use_bias_corr[i][0,1].item()

            ax = axes[i, 0]
            im = ax.imshow(mean_slice_use[i].cpu().numpy(), cmap='viridis', aspect='auto', vmin=0, vmax=3)
            ax.set_title(f'Layer {i}: underused: {underused[i]:.1f}% / overused: {overused[i]:.1f}%')
            ax.set_xlabel('Slice Index')
            ax.set_ylabel('Head Index')
            fig.colorbar(im, ax=ax)
            im.cmap.set_over('red')
            im.cmap.set_under('blue')
            
            ax = axes[i, 1]
            im = ax.imshow(bias[i].cpu().numpy(), cmap='viridis', aspect='auto', vmin=-1, vmax=1)
            ax.set_title(f'Layer {i}: bias (min: {bias[i].min().item():.1e}, max: {bias[i].max().item():.1e}, corr: {corr:.3f})')
            ax.set_xlabel('Slice Index')
            ax.set_ylabel('Head Index')
            fig.colorbar(im, ax=ax)
            im.cmap.set_over('red')
            im.cmap.set_under('blue')

        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, 'utilization.png'))
        plt.savefig(os.path.join(ckpt_dir, '..', 'utilization.png'))

        return

#======================================================================#
#