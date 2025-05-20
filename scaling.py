#
import os
import time
import shutil
import subprocess
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import seaborn as sns
from bench.models.cat import ClusterAttentionBlock

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1000  # in thousands

def measure_memory_time(block, x, num_steps=10):
    # Warmup
    for _ in range(10):
        _ = block(x)
    torch.cuda.synchronize()

    y = torch.rand_like(x)
    lossfun = torch.nn.MSELoss()

    # Measure memory
    torch.cuda.reset_peak_memory_stats()
    yh = block(x)
    loss = lossfun(yh, y)
    loss.backward()
    torch.cuda.synchronize()
    memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MiB
    
    # Measure time
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_steps):
        yh = block(x)
        loss = lossfun(yh, y)
        loss.backward()
    torch.cuda.synchronize()
    time_taken = (time.time() - start_time) * 1000  # Convert to ms

    return memory, time_taken

def collect_data():
    # Test configurations
    num_points = [1024]  # Only N=1024
    channel_dim = [64, 128, 256]#, 512, 1024]
    num_clusters = [8, 16, 32, 64, 128, 256, 512]
    num_heads = [1, 2, 4, 8, 16, 32]
    
    # Create empty list to store results
    results = []
    
    for C in channel_dim:
        for M in num_clusters:
            for H in num_heads:
                # Skip invalid configurations
                if (C % H != 0) or (H > C // 2):
                    continue

                print(f'Case: C={C}, M={M}, H={H}')

                # Create block
                block = ClusterAttentionBlock(
                    num_heads=H,
                    channel_dim=C,
                    num_clusters=M,
                    num_projection_heads=H,
                    num_latent_blocks=1,
                    if_latent_mlp=False,
                    if_pointwise_mlp=True,
                    cluster_head_mixing=True,
                ).cuda()
                
                num_params = count_parameters(block)
                
                for N in num_points:
                    # Create input tensor
                    x = torch.randn(1, N, C).cuda()
                    
                    # Measure memory and time
                    memory, time_taken = measure_memory_time(block, x)
                    
                    # Store results
                    results.append({
                        'N': N,
                        'C': C,
                        'M': M,
                        'H': H,
                        'Hp': H,
                        'num_params': num_params,
                        'memory_mib': memory,
                        'time_ms': time_taken
                    })
                    
                    # Free memory
                    torch.cuda.empty_cache()
                
                del block
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv('memory_stats.csv', index=False)
    return df

def plot_results(df):
    # Filter for N=1024
    df = df[df['N'] == 1024]
    
    # Create figure with three subplots
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Colors for different configurations
    num_clusters = sorted(df['M'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(num_clusters)))
    
    # Plot 1: Memory vs Channel Dim
    for i, M in enumerate(num_clusters):
        for H in sorted(df['H'].unique()):
            mask = (df['M'] == M) & (df['H'] == H)
            if not mask.any():
                continue
            
            config_data = df[mask]
            num_params = config_data['num_params'].iloc[0]
            
            label = f'CAT Block [M={M}, H={H}] ({num_params:.1f}k params)'
            ax1.loglog(config_data['C'], config_data['memory_mib'], color=colors[i], label=label)
    
    # Plot 2: Time vs Channel Dim
    for i, M in enumerate(num_clusters):
        for H in sorted(df['H'].unique()):
            mask = (df['M'] == M) & (df['H'] == H)
            if not mask.any():
                continue
            
            config_data = df[mask]
            ax2.loglog(config_data['C'], config_data['time_ms'], color=colors[i])
    
    # Plot 3: Heatmap
    # Create pivot tables for memory and time
    memory_pivot = df.pivot_table(
        values='memory_mib', 
        index='M', 
        columns='C', 
        aggfunc='mean'
    )
    time_pivot = df.pivot_table(
        values='time_ms', 
        index='M', 
        columns='C', 
        aggfunc='mean'
    )
    
    # Plot heatmap
    sns.heatmap(memory_pivot, annot=True, fmt='.0f', cmap='viridis', ax=ax3)
    
    # Add text annotations for time
    for i in range(len(memory_pivot.index)):
        for j in range(len(memory_pivot.columns)):
            memory = memory_pivot.iloc[i, j]
            time = time_pivot.iloc[i, j]
            if not np.isnan(memory):
                ax3.text(j + 0.5, i + 0.5, f'\n{time:.0f}ms', 
                        ha='center', va='center', color='white')
    
    # Customize plots
    ax1.set_xlabel('Channel Dimension (C)')
    ax1.set_ylabel('Memory Footprint (MiB)')
    ax1.set_title('Memory Scaling')
    ax1.grid(True)
    
    ax2.set_xlabel('Channel Dimension (C)')
    ax2.set_ylabel('Time for 10 GD Steps (ms)')
    ax2.set_title('Time Scaling')
    ax2.grid(True)
    
    ax3.set_xlabel('Channel Dimension (C)')
    ax3.set_ylabel('Number of Clusters (M)')
    ax3.set_title('Memory (MiB) / Time (ms) Heatmap')
    
    # Set x-axis ticks to powers of 2
    for ax in [ax1, ax2]:
        x_ticks = sorted(df['C'].unique())
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'$2^{{{int(np.log2(x))}}}$' for x in x_ticks])
    
    # Add legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05),
              ncol=3, frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for legend
    plt.savefig('cat_scaling.png', bbox_inches='tight', dpi=300)
    plt.close()

def memory_time_analysis():
    # Load or collect data
    if args.force_reload or not pd.io.common.file_exists('memory_stats.csv'):
        print("Collecting data...")
        df = collect_data()
    else:
        print("Loading data from CSV...")
        df = pd.read_csv('memory_stats.csv')

    # Plot results
    print("Generating plots...")
    plot_results(df)
    print("Done! Results saved to cat_scaling.png")
    
def scaling_study(dataset: str, gpu_count: int = None, max_jobs_per_gpu: int = 2):
    if gpu_count is None:
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
    for if_latent_mlp in [False, True]:
        for if_pointwise_mlp in [True, False]:
            for cluster_head_mixing in [True, False]:
                for channel_dim in [64, 128, 256, 512]:
                    for num_clusters in [16, 32, 64, 128, 256, 512]:
                        for num_blocks in [1, 2, 4, 8]:
                            for num_latent_blocks in [0, 1, 2, 4, 8]:
                                for num_projection_heads in [1, 2, 4, 8, 16, 32]:
                                    for num_heads in [4, 8, 16, 32]:

                                        # Skip invalid configurations
                                        if (channel_dim % num_heads != 0) or (num_heads > channel_dim // 16):
                                            continue
                                        if (channel_dim % num_projection_heads != 0) or (num_projection_heads > channel_dim // 8):
                                            continue

                                        #------------------------------------#
                                        # EXP 1: C = 64
                                        #------------------------------------#
                                        if channel_dim not in [64]:
                                            continue
                                        if num_clusters not in [32, 64, 128]:
                                            continue
                                        if num_projection_heads not in [1, 2, 4, 8]:
                                            continue
                                        # #------------------------------------#
                                        # # EXP 2: C = 128
                                        # #------------------------------------#
                                        # if channel_dim not in [128]:
                                        #     continue
                                        # if num_clusters not in [32, 64, 128]:
                                        #     continue
                                        # if num_projection_heads not in [1, 2, 4, 8]:
                                        #     continue
                                        # #------------------------------------#
                                        # # EXP 3: C = 256
                                        # #------------------------------------#
                                        # if channel_dim not in [256]:
                                        #     continue
                                        # if num_clusters not in [32, 64, 128, 256]:
                                        #     continue
                                        # if num_projection_heads not in [1, 2, 4, 8]:
                                        #     continue

                                        #------------------------------------#
                                        exp_name = f'scaling_{dataset}_MLPL_{if_latent_mlp}_MLPP_{if_pointwise_mlp}_MIX_{cluster_head_mixing}_C_{channel_dim}_M_{num_clusters}_B_{num_blocks}_LB_{num_latent_blocks}_HP_{num_projection_heads}_H_{num_heads}'
                                        exp_name = os.path.join(f'scaling_{dataset}', exp_name)
                                        
                                        case_dir = os.path.join('.', 'out', 'bench', exp_name)
                                        if os.path.exists(case_dir):
                                            if os.path.exists(os.path.join(case_dir, 'ckpt10', 'rel_error.json')):
                                                continue
                                            else:
                                                print(f"Experiment {exp_name} exists but ckpt10/rel_error.json does not exist. Removing failed experiment and re-running.")
                                                shutil.rmtree(case_dir)

                                        job_queue.append({
                                            'if_latent_mlp': if_latent_mlp,
                                            'if_pointwise_mlp': if_pointwise_mlp,
                                            'channel_dim': channel_dim,
                                            'num_blocks': num_blocks,
                                            'num_latent_blocks': num_latent_blocks,
                                            'num_projection_heads': num_projection_heads,
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

            # print(f"Running job {jobid} / {njobs} on GPU {gpuid}")

            job = job_queue.pop(0)
            process = subprocess.Popen([
                    'python', '-m', 'bench',
                    '--exp_name', job['exp_name'],
                    '--train', str('True'),
                    '--model_type', str(1),
                    '--dataset', dataset,
                    # training arguments
                    '--epochs', str(epochs),
                    '--weight_decay', str(1e-5),
                    '--batch_size', str(batch_size),
                    # model arguments
                    '--if_latent_mlp', str(job['if_latent_mlp']),
                    '--if_pointwise_mlp', str(job['if_pointwise_mlp']),
                    '--channel_dim', str(job['channel_dim']),
                    '--num_blocks', str(job['num_blocks']),
                    '--num_latent_blocks', str(job['num_latent_blocks']),
                    '--num_projection_heads', str(job['num_projection_heads']),
                    '--num_heads', str(job['num_heads']),
                    '--num_clusters', str(job['num_clusters']),
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            active_processes[gpuid].append(process)
            jobid += 1
            
            pbar.update(1)

        # Wait 5 mins and check for completed jobs
        time.sleep(300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CAT model scaling study')
    parser.add_argument('--gpu-count', type=int, default=None, help='Number of GPUs to use')
    parser.add_argument('--dataset', type=str, default='elasticity', help='Dataset to use')
    args = parser.parse_args()
    
    # memory_time_analysis() 
    scaling_study(args.dataset, args.gpu_count)
