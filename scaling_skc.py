#
import os
import time
import shutil
import subprocess
import json, yaml
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import LogNorm

import numpy as np
import pandas as pd
import argparse
import seaborn as sns

#======================================================================#
def collect_scaling_study_data(dataset: str):
    data_dir = os.path.join('.', 'out', 'bench', f'scaling_skc_{dataset}')

    # Initialize empty dataframe
    df = pd.DataFrame()

    # Check if case directory exists
    if os.path.exists(data_dir):
        # Get all subdirectories (each represents a case)
        cases = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        for case in cases:
            case_path = os.path.join(data_dir, case)
            
            if not os.path.exists(os.path.join(case_path, 'config.yaml')):
                continue
            if not os.path.exists(os.path.join(case_path, 'num_params.txt')):
                continue

            # Initialize case data dictionary
            case_data = {}
            
            # Check for and load relative error data
            rel_error_path = os.path.join(case_path, 'ckpt10', 'rel_error.json')
            if os.path.exists(rel_error_path):
                with open(rel_error_path, 'r') as f:
                    rel_error = json.load(f)
                case_data.update({
                    'train_rel_error': rel_error.get('train_rel_error'),
                    'test_rel_error': rel_error.get('test_rel_error')
                })
            
            # Load config data
            config_path = os.path.join(case_path, 'config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                case_data.update({
                    'channel_dim': config.get('channel_dim'),
                    'num_clusters': config.get('num_clusters'),
                    'num_blocks': config.get('num_blocks'),
                    'num_heads': config.get('num_heads'),
                    'cluster_head_mixing': config.get('cluster_head_mixing'),
                })

            # Load num_params
            num_params_path = os.path.join(case_path, 'num_params.txt')
            if os.path.exists(num_params_path):
                with open(num_params_path, 'r') as f:
                    num_params = int(f.read().strip())
                case_data.update({'num_params': num_params})
            
            # Add case data to dataframe
            df = pd.concat([df, pd.DataFrame([case_data])], ignore_index=True)

            df['head_dim'] = df['channel_dim'] // df['num_heads']

        print(f"Collected {len(df)} cases for {dataset} dataset.")

    return df

def plot_scaling_study_results(dataset: str, df: pd.DataFrame):

    output_dir = os.path.join('.', 'out', 'bench', f'scaling_skc_{dataset}_analysis')
    os.makedirs(output_dir, exist_ok=True)

    #---------------------------------------------------------#
    # HEATMAP across Blocks vs Latent Blocks
    #---------------------------------------------------------
    cmap = 'RdYlBu_r'
    vmin, vmax = (1e-2, 1e-1) if dataset in ['shapenet_car'] else (1e-3, 1e-2)

    configs = df[['cluster_head_mixing', 'channel_dim', 'num_clusters', 'num_heads']].drop_duplicates()

    print(f"Found {len(configs)} unique configurations for B vs BL heatmap.")

    for _, config in configs.iterrows():

        if_latent_mlp = config['if_latent_mlp']
        if_pointwise_mlp = config['if_pointwise_mlp']
        cluster_head_mixing = config['cluster_head_mixing']
        channel_dim = config['channel_dim']
        num_clusters = config['num_clusters']
        num_heads = config['num_heads']
        num_heads = config['num_heads']

        df_ = df[
            (df['cluster_head_mixing'] == cluster_head_mixing) &
            (df['channel_dim'] == channel_dim) &
            (df['num_clusters'] == num_clusters) &
            (df['num_heads'] == num_heads)
        ]

        name_str = f'MIX_{cluster_head_mixing}_C_{channel_dim}_M_{num_clusters}_HP_{num_heads}'
        title_str = f'Cluster Head Mixing: {cluster_head_mixing}, Channel Dim: {channel_dim}, # Clusters: {num_clusters}, # Projection Heads: {num_heads}'

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle(title_str)

        # Create pivot tables with numeric values for coloring and annotations
        pivot_test = df_.pivot_table(
            values='test_rel_error',
            columns='num_blocks',
            index='num_latent_blocks',
            aggfunc='mean'
        )
        pivot_train = df_.pivot_table(
            values='train_rel_error',
            columns='num_blocks',
            index='num_latent_blocks',
            aggfunc='mean'
        )
        pivot_params = df_.pivot_table(
            values='num_params',
            columns='num_blocks',
            index='num_latent_blocks',
            aggfunc='mean'
        )

        if pivot_train.empty or pivot_test.empty:
            plt.close()
            continue
        # annot_params = pivot_params.map(lambda x: format_param(x))

        # Create combined annotations
        combined_annot = pd.DataFrame(index=pivot_test.index, columns=pivot_test.columns, dtype=str)
        for i in range(combined_annot.shape[0]):
            for j in range(combined_annot.shape[1]):
                train_val = format_sci(pivot_train.iloc[i, j])
                test_val = format_sci(pivot_test.iloc[i, j])
                params_val = format_param(pivot_params.iloc[i, j])
                combined_annot.iloc[i, j] = f"{train_val}\n{test_val}\n{params_val}"

        annot_kws = {"size": 11, "weight": "bold"}
        linear_scale_kw = {'vmin': vmin, 'vmax': vmax}

        heatmap = sns.heatmap(pivot_test, annot=combined_annot, fmt='', cmap=cmap, ax=ax, **linear_scale_kw, annot_kws=annot_kws, linewidths=0.5, linecolor='black')

        # Format colorbar ticks in scientific notation
        cbar = heatmap.collections[0].colorbar
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        cbar.set_label('Test relative error')

        ax.set_title('Train relative error/ test relative error/ parameter count')
        ax.set_ylabel('Number of latent blocks (BL)')
        ax.set_xlabel('Number of blocks (B)')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'heatmap_L_vs_LB_{name_str}.png'))
        plt.close()

    #---------------------------------------------------------#
    # LINEPLOT of test error vs projection head dim
    #---------------------------------------------------------
    
    df_ = df
    df_ = df_[df_['cluster_head_mixing'] == True]
    df_ = df_[df_['num_blocks'].isin([8])]
    df_ = df_[df_['num_latent_blocks'].isin([0,1])]

    configs = df_[['cluster_head_mixing', 'channel_dim', 'num_clusters', 'num_heads']].drop_duplicates()

    print(f"Found {len(configs)} unique configurations for num projection head lineplot.")

    configs = configs.sort_values(by=['channel_dim', 'num_clusters', 'num_blocks', 'num_latent_blocks'])

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.set_xlabel('Projection head dimension')
    ax1.set_ylabel('Test relative error')
    ax1.set_title(f'Test relative error vs projection head dimension')

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.set_xlabel('Number of projection heads')
    ax2.set_ylabel('Test relative error')
    ax2.set_title(f'Test relative error vs number of projection heads')

    for _, config in configs.iterrows():

        cluster_head_mixing = config['cluster_head_mixing']
        channel_dim = config['channel_dim']
        num_blocks = config['num_blocks']
        num_clusters = config['num_clusters']

        label = f'c={channel_dim}, m={num_clusters}, b={num_blocks}, hp={num_heads}'

        df__ = df_[
            (df_['channel_dim'] == channel_dim) &
            (df_['num_blocks'] == num_blocks) &
            (df_['num_clusters'] == num_clusters) &
            (df_['num_heads'] == num_heads)
        ]

        df__ = df__.sort_values(by='head_dim')

        if df__.empty:
            continue

        linestyle=':' if cluster_head_mixing else '-'
        ax1.plot(df__['head_dim'], df__['test_rel_error'], label=label, marker='o', linestyle=linestyle)
        ax2.plot(df__['num_heads'], df__['test_rel_error'], label=label, marker='o', linestyle=linestyle)

    ax1.legend()
    ax2.legend()
    fig1.savefig(os.path.join(output_dir, f'lineplot_head_dim.png'))
    fig2.savefig(os.path.join(output_dir, f'lineplot_num_heads.png'))
    fig1.close()
    fig2.close()

    #---------------------------------------------------------#
    return

#======================================================================#
def format_param(x):
    if x > 1e6:
        return f"{x/1e6:.1f}m"
    elif x > 1e3:
        return f"{x/1e3:.1f}k"
    else:
        return f"{x:.1f}"

def format_sci(x):
    if pd.isna(x):
        return ""
    return f"{x:.2e}".replace("e+0", "e+").replace("e-0", "e-")

#======================================================================#
def eval_scaling_study(dataset: str):
    df = collect_scaling_study_data(dataset)
    plot_scaling_study_results(dataset, df)
    return

#======================================================================#
def train_scaling_study(dataset: str, gpu_count: int = None, max_jobs_per_gpu: int = 2):
    if gpu_count is None:
        import torch
        gpu_count = torch.cuda.device_count()
    if dataset == 'elasticity':
        epochs = 500
        batch_size = 2
    elif dataset == 'shapenet_car':
        epochs = 200
        batch_size = 1
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    print(f"Using {gpu_count} GPUs to run scaling study on {dataset} dataset.")

    # Create a queue of all jobs
    job_queue = []
    for cluster_head_mixing in [True, False]:
        for channel_dim in [64, 128]:
            for num_clusters in [8, 16, 32, 64, 128]:
                for num_blocks in [1, 2, 4, 8]:
                    for num_heads in [1, 2, 4, 8, 16, 32]:

                                # Skip invalid configurations
                                if num_heads == 0:
                                    continue
                                if channel_dim % num_heads != 0:
                                    continue

                                head_dim = channel_dim // num_heads

                                # ensure head dim is at least 4
                                if head_dim < 4:
                                    continue

                                #------------------------------------#
                                # EXPERIMENT CRITERIA
                                #------------------------------------#
                                #------------------------------------#

                                exp_name = f'scaling_{dataset}_MIX_{cluster_head_mixing}_C_{channel_dim}_M_{num_clusters}_B_{num_blocks}_HP_{num_heads}'
                                exp_name = os.path.join(f'scaling_skc_{dataset}', exp_name)
                                
                                case_dir = os.path.join('.', 'out', 'bench', exp_name)
                                if os.path.exists(case_dir):
                                    if os.path.exists(os.path.join(case_dir, 'ckpt10', 'rel_error.json')):
                                        print(f"Experiment {exp_name} exists. Skipping.")
                                        continue
                                    else:
                                        print(f"Experiment {exp_name} exists but ckpt10/rel_error.json does not exist. Removing and re-running.")
                                        shutil.rmtree(case_dir)

                                job_queue.append({
                                    'channel_dim': channel_dim,
                                    'num_blocks': num_blocks,
                                    'num_heads': num_heads,
                                    'num_clusters': num_clusters,
                                    'exp_name': exp_name
                                })

    jobid = 0
    njobs = len(job_queue)
    
    print(f"Running {njobs} jobs on {gpu_count} GPUs.")
    pbar = tqdm(total=njobs, desc="Running jobs", ncols=80)

    # Run jobs
    active_processes = [[] for _ in range(gpu_count)]
    while job_queue or any(len(p) > 0 for p in active_processes):

        # Check completed processes
        for i in range(gpu_count):
            # p.poll() returns None if the process is still running
            for p in active_processes[i]:
                if p.poll() is not None:
                    if p.returncode != 0:
                        print(f"\nExperiment {p.args[4]} failed on GPU {i}. Removing and re-running.")
                        # remove failed experiment and re-run
                        case_dir = os.path.join('.', 'out', 'bench', p.args[4])
                        shutil.rmtree(case_dir)
                        job_queue.append(job)
                        pbar.update(-1)
                        jobid -= 1

            # Remove completed processes
            active_processes[i] = [p for p in active_processes[i] if p.poll() is None]

        # Start new jobs on available GPUs
        while any(len(p) < max_jobs_per_gpu for p in active_processes) and job_queue:
            gpuid = min(range(gpu_count), key=lambda i: len(active_processes[i]))
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)

            job = job_queue.pop(0)
            process = subprocess.Popen([
                    'python', '-m', 'bench',
                    '--exp_name', job['exp_name'],
                    '--train', str('True'),
                    '--model_type', str(2),
                    '--dataset', dataset,
                    # training arguments
                    '--epochs', str(epochs),
                    '--weight_decay', str(1e-5),
                    '--batch_size', str(batch_size),
                    # model arguments
                    '--channel_dim', str(job['channel_dim']),
                    '--num_blocks', str(job['num_blocks']),
                    '--num_heads', str(job['num_heads']),
                    '--num_clusters', str(job['num_clusters']),
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            active_processes[gpuid].append(process)
            jobid += 1
            
            pbar.update(1)

        # Wait 5 mins and check for completed jobs
        time.sleep(300)
    return

def clean_scaling_study(dataset: str):
    output_dir = os.path.join('.', 'out', 'bench', f'scaling_skc_{dataset}')
    for case_name in [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]:
        case_dir = os.path.join(output_dir, case_name)
        if os.path.exists(os.path.join(case_dir, 'ckpt10', 'rel_error.json')):
            for ckpt in [f'ckpt{i:02d}' for i in range(10)]:
                if os.path.exists(os.path.join(case_dir, ckpt)):
                    shutil.rmtree(os.path.join(case_dir, ckpt))
            if os.path.exists(os.path.join(case_dir, 'ckpt10', 'model.pt')):
                os.remove(os.path.join(case_dir, 'ckpt10', 'model.pt'))
        else:
            shutil.rmtree(case_dir)
    return

#======================================================================#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SkinnyCAT model scaling study')

    parser.add_argument('--eval', type=bool, default=False, help='Evaluate scaling study results')
    parser.add_argument('--train', type=bool, default=False, help='Train scaling study')
    parser.add_argument('--clean', type=bool, default=False, help='Clean scaling study results')

    parser.add_argument('--dataset', type=str, default='elasticity', help='Dataset to use')
    parser.add_argument('--gpu-count', type=int, default=None, help='Number of GPUs to use')
    parser.add_argument('--max-jobs-per-gpu', type=int, default=2, help='Maximum number of jobs per GPU')

    args = parser.parse_args()

    if args.train:
        train_scaling_study(args.dataset, args.gpu_count, args.max_jobs_per_gpu)
    if args.eval:
        eval_scaling_study(args.dataset)
    if args.clean:
        clean_scaling_study(args.dataset)
        
    if not args.train and not args.eval and not args.clean:
        print("No action specified. Please specify either --train or --eval or --clean.")

    exit()

#======================================================================#
#