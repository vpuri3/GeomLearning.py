#
import os

from sympy import print_fcode
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
        if trainer.GLOBAL_RANK == 0:
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
                if trainer.GLOBAL_RANK == 0:
                    print(f"No {split} dataset.")
                continue

            if self.final:
                split_dir = os.path.join(ckpt_dir, f'vis_{split}')
                os.makedirs(split_dir, exist_ok=True)
            
            stats_file = os.path.join(ckpt_dir, f'{split}_stats.txt')

            # distribute cases across ranks
            num_cases = len(dataset)
            cases_per_rank = num_cases // trainer.WORLD_SIZE 
            icase0 = trainer.GLOBAL_RANK * cases_per_rank
            icase1 = (trainer.GLOBAL_RANK + 1) * cases_per_rank if trainer.GLOBAL_RANK != trainer.WORLD_SIZE - 1 else num_cases

            max_eval_cases = self.num_eval_cases // trainer.WORLD_SIZE

            case_nums = []
            case_names = []
            l2s = []
            r2s = []

            if trainer.GLOBAL_RANK == 0:
                pbar = tqdm(total=num_cases, desc=f"Evaluating {split} dataset", ncols=80)
            
            for icase in range(icase0, icase1):
                data = dataset[icase].to(device)
                data.yh = model(data)
                data.e = data.y - data.yh
                data.yp = data.yh * transform.scale.to(device)

                case_nums.append(icase)
                case_names.append(data.metadata['case_name'])
                l2s.append(torch.nn.MSELoss()(data.yh, data.y).item())
                r2s.append(mlutils.r2(data.yh, data.y))

                if self.final and (len(case_nums) < max_eval_cases):
                    base_name = os.path.basename(self.case_dir)
                    case_name = data.metadata["case_name"]
                    file_name = f'{base_name}-{split}{str(icase).zfill(4)}-{case_name}'
                    out_file = os.path.join(split_dir, file_name + '.vtu')
                    visualize_pyv(data, out_file)

                del data
                
                if trainer.GLOBAL_RANK == 0:
                    pbar.update(trainer.WORLD_SIZE)
            
            if trainer.GLOBAL_RANK == 0:
                pbar.close()

            df = pd.DataFrame({
                'case_num': case_nums,
                'case_name': case_names,
                'MSE': l2s,
                'R-Square': r2s
            })
            
            # gather dataframe across ranks
            df = vstack_dataframes_across_ranks(df, trainer)

            if trainer.GLOBAL_RANK == 0:
                print(f"Saving {split} stats to {stats_file}")
                df.to_csv(stats_file, index=False)
        
        if trainer.DDP:
            torch.distributed.barrier()
            
        r2_values = {'train': [], 'test': []}
        for split in ['train', 'test']:
            stats_file = os.path.join(ckpt_dir, f'{split}_stats.txt')
            df = pd.read_csv(stats_file)
            r2_values[split] = df['R-Square'].values

        if trainer.GLOBAL_RANK == 0:
            print(f"Plotting R-Squared boxplot to {ckpt_dir}/r2_boxplot.png")
            r2_boxplot(r2_values, filename=os.path.join(ckpt_dir, 'r2_boxplot.png'))

        return

#======================================================================#
class TimeseriesCallback(Callback):
    def __init__(
        self, case_dir: str, mesh: bool, save_every=None, num_eval_cases=None,
        autoreg_start=1,
    ):
        super().__init__(case_dir, mesh=mesh, save_every=save_every, num_eval_cases=num_eval_cases)
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
                if trainer.GLOBAL_RANK == 0:
                    print(f"No {split} dataset.")
                continue
            
            split_dir = os.path.join(ckpt_dir, f'vis_{split}')

            if trainer.GLOBAL_RANK == 0:
                os.makedirs(split_dir, exist_ok=True)
            
            # distribute cases across ranks
            num_cases = len(dataset.case_files)
            cases_per_rank = num_cases // trainer.WORLD_SIZE 
            icase0 = trainer.GLOBAL_RANK * cases_per_rank
            icase1 = (trainer.GLOBAL_RANK + 1) * cases_per_rank if trainer.GLOBAL_RANK != trainer.WORLD_SIZE - 1 else num_cases
            
            max_eval_cases = self.num_eval_cases // trainer.WORLD_SIZE

            case_nums = []
            case_names = []

            l2_ARs = []
            l2_NRs = []

            r2_ARs = []
            r2_NRs = []

            if trainer.GLOBAL_RANK == 0:
                pbar = tqdm(total=num_cases, desc=f"Evaluating {split} dataset")
            
            for icase in range(icase0, icase1):
                case_idx = dataset.case_range(icase)
                case_data = dataset[case_idx]
                case_name = case_data[0].metadata['case_name']

                # if len(case_nums) > max_eval_cases:
                #     break

                case_nums.append(icase)
                case_names.append(case_name)
                
                for (autoreg, ext) in zip([True, False], ['AR', 'NR']):
                    eval_data, l2s, r2s = march_case(
                        model, case_data, transform,
                        autoreg=autoreg, device=device, K=self.autoreg_start,
                    )

                    case_dir = os.path.join(split_dir, f"{split}{str(icase).zfill(3)}-{ext}-{case_name}")
                    file_name = f'{os.path.basename(self.case_dir)}-{split}{str(icase).zfill(4)}-{ext}-{case_name}'

                    # if self.final and len(case_nums) < self.num_eval_cases:
                    #     visualize_timeseries_pyv(eval_data, case_dir, merge=True, name=file_name)

                    if ext == 'AR':
                        l2_ARs.append(l2s)
                        r2_ARs.append(r2s)
                    else:
                        l2_NRs.append(l2s)
                        r2_NRs.append(r2s)

                    del eval_data 

                del case_data
                
                if trainer.GLOBAL_RANK == 0:
                    pbar.update(trainer.WORLD_SIZE)
                    
            if trainer.GLOBAL_RANK == 0:
                pbar.close()

            # Convert list of stats arrays into a DataFrame where each row represents
            # a time step and each column represents a case
            df_l2_AR = pd.DataFrame(l2_ARs).transpose()
            df_r2_AR = pd.DataFrame(r2_ARs).transpose()
            df_l2_NR = pd.DataFrame(l2_NRs).transpose()
            df_r2_NR = pd.DataFrame(r2_NRs).transpose()

            # Assign case numbers as column names
            df_l2_AR.columns = case_nums
            df_r2_AR.columns = case_nums
            df_l2_NR.columns = case_nums
            df_r2_NR.columns = case_nums

            # Assign step numbers as index
            df_l2_AR.index.name = 'Step'
            df_r2_AR.index.name = 'Step'
            df_l2_NR.index.name = 'Step'
            df_r2_NR.index.name = 'Step'

            # create dataframe for each autoreg
            df_l2_AR = hstack_dataframes_across_ranks(df_l2_AR, trainer)
            df_r2_AR = hstack_dataframes_across_ranks(df_r2_AR, trainer)
            df_l2_NR = hstack_dataframes_across_ranks(df_l2_NR, trainer)
            df_r2_NR = hstack_dataframes_across_ranks(df_r2_NR, trainer)
            
            if trainer.GLOBAL_RANK == 0:
                print(f"Saving {split} statistics to {ckpt_dir}")
                df_l2_AR.to_csv(os.path.join(ckpt_dir, f'l2_AR_stats_{split}.txt'), index=False)
                df_r2_AR.to_csv(os.path.join(ckpt_dir, f'r2_AR_stats_{split}.txt'), index=False)
                df_l2_NR.to_csv(os.path.join(ckpt_dir, f'l2_NR_stats_{split}.txt'), index=False)
                df_r2_NR.to_csv(os.path.join(ckpt_dir, f'r2_NR_stats_{split}.txt'), index=False)
            
        if trainer.DDP:
            torch.distributed.barrier()

        # make plots
        for split in ['train', 'test']:
            # df_l2_AR = pd.read_csv(os.path.join(ckpt_dir, f'l2_AR_stats_{split}.txt'))
            # df_l2_NR = pd.read_csv(os.path.join(ckpt_dir, f'l2_NR_stats_{split}.txt'))
            df_r2_AR = pd.read_csv(os.path.join(ckpt_dir, f'r2_AR_stats_{split}.txt'))
            # df_r2_NR = pd.read_csv(os.path.join(ckpt_dir, f'r2_NR_stats_{split}.txt'))

            if trainer.GLOBAL_RANK == 0:
                print(f"Saving R-Squared plots to {ckpt_dir}/r2_plot_{split}.png")
                r2_timeseries(df_r2_AR, filename=os.path.join(ckpt_dir, f'r2_plot_{split}.png'))

        return

#======================================================================#
# Plotting functions
#======================================================================#
def r2_timeseries(df, filename=None, dpi=175,):
    plt.figure(figsize=(8, 4), dpi=dpi)
    plt.ylim(-1, 1)
    
    medians = df.median(axis=1)
    q1 = df.quantile(0.25, axis=1)
    q3 = df.quantile(0.75, axis=1)
    tstep = np.arange(len(medians))
    
    plt.plot(tstep, medians, color='k', label='Median')
    plt.fill_between(
        tstep, q1, q3,
        color='k', alpha=0.2,
        label='Middle 50%',
    )
    
    plt.xlabel('Time Step')
    plt.ylabel('R-Squared')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot if filename is provided
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    return

def r2_boxplot(
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
# Combine DataFrames across distributed processes
#======================================================================#
def hstack_dataframes_across_ranks(df: pd.DataFrame, trainer: mlutils.Trainer) -> pd.DataFrame:
    """
    Combine DataFrames across distributed processes horizontally by adding columns.
    
    Args:
        df: Local DataFrame to combine
        trainer: Trainer object containing distributed training info
        
    Returns:
        Combined DataFrame with columns from all processes
    """
    if not trainer.DDP:
        return df
        
    local_data = df.to_dict('list')  # Get columns as lists
    
    # Gather data from all processes
    gathered_data = [None] * trainer.WORLD_SIZE
    torch.distributed.all_gather_object(gathered_data, local_data)
    
    # Combine columns from all processes
    combined_data = {}
    for rank_data in gathered_data:
        combined_data.update(rank_data)
        
    return pd.DataFrame(combined_data)

def vstack_dataframes_across_ranks(df: pd.DataFrame, trainer: mlutils.Trainer) -> pd.DataFrame:
    """
    Combine DataFrames across distributed processes vertically by adding rows.
    
    Args:
        df: Local DataFrame to combine
        trainer: Trainer object containing distributed training info
        
    Returns:
        Combined DataFrame from all processes
    """
    if not trainer.DDP:
        return df
        
    local_data = df.to_dict('records')
    
    # Gather data from all processes
    gathered_data = [None] * trainer.WORLD_SIZE
    torch.distributed.all_gather_object(gathered_data, local_data)
    
    # Flatten the list of lists and create final DataFrame
    all_data = [item for sublist in gathered_data for item in sublist]
    return pd.DataFrame(all_data)

#======================================================================#
