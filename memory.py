import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
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
    channel_dim = [64,]# 128, 256, 512, 1024]
    num_clusters = [8, 16, 32, 64, 128, 256,]# 512]
    num_heads = [1, 2, 4, 8, 16, 32,]# 64, 128]
    num_points = [256, 512, 1024, 2048, 4096, 8192,]# 16384, 32768, 65536]
    
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
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Colors for different configurations
    num_clusters = sorted(df['M'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(num_clusters)))
    
    for i, M in enumerate(num_clusters):
        for H in sorted(df['H'].unique()):
            for C in sorted(df['C'].unique()):
                # Filter data for this configuration
                mask = (df['M'] == M) & (df['H'] == H) & (df['C'] == C)
                if not mask.any():
                    continue
                
                config_data = df[mask]
                num_params = config_data['num_params'].iloc[0]
                
                # Plot results
                label = f'CAT Block [C={C}, M={M}, Hp={H}, H={H}] ({num_params:.1f}k params)'
                ax1.loglog(config_data['N'], config_data['memory_mib'], color=colors[i], label=label)
                ax2.loglog(config_data['N'], config_data['time_ms'], color=colors[i], label=label)
    
    # Customize plots
    for ax in [ax1, ax2]:
        ax.set_xlabel('Number of Input Points (N)')
        ax.grid(True)
        
        # Set x-axis ticks to powers of 2
        x_ticks = sorted(df['N'].unique())
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'$2^{{{int(np.log2(x))}}}$' for x in x_ticks])
    
    ax1.set_ylabel('Memory Footprint (MiB)')
    ax1.set_title('Memory Scaling')
    
    ax2.set_ylabel('Time for 10 GD Steps (ms)')
    ax2.set_title('Time Scaling')
    
    # Add legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5))
    
    plt.tight_layout()
    plt.savefig('cat_scaling.png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='CAT Block Scaling Analysis')
    parser.add_argument('--force-reload', action='store_true', 
                      help='Force reload data even if CSV exists')
    args = parser.parse_args()
    
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

if __name__ == '__main__':
    main() 