import os
import json
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import argparse

def parse_dir_name(dirname):
    """Extract hyperparameters from directory name."""
    pattern = r'L_(\d+)_B_(\d+)(?:_HP_(\d+))?(?:_M_(\d+))?(?:_mix_([tf]))?'
    match = re.search(pattern, dirname)
    if match:
        L, B, HP, M, mix = match.groups()
        return {
            'L': int(L),
            'B': int(B),
            'HP': int(HP) if HP else 8,  # Default to 8 if not specified
            'M': int(M) if M else 64,  # Default to 64 if not specified
            'mix': mix if mix else 't'  # Default to True if not specified
        }
    return None

def collect_results(base_dir):
    """Collect results from all experiment directories."""
    results = []
    
    for dirname in os.listdir(base_dir):
        if not os.path.isdir(os.path.join(base_dir, dirname)):
            continue

        params = parse_dir_name(dirname)
        if not params:
            continue

        json_path = os.path.join(base_dir, dirname, 'ckpt10', 'rel_error.json')
        if not os.path.exists(json_path):
            print(f"Error: File {json_path} does not exist")
            continue

        try:
            with open(json_path, 'r') as f:
                errors = json.load(f)
                
            if 'train_loss' in errors and 'train_rel_error' not in errors:
                errors['train_rel_error'] = errors['train_loss']
            if 'test_loss' in errors and 'test_rel_error' not in errors:
                errors['test_rel_error'] = errors['test_loss']

            results.append({
                **params,
                'train_rel_error': errors.get('train_rel_error', float('nan')),
                'test_rel_error': errors.get('test_rel_error', float('nan'))
            })
        except Exception as e:
            print(f"Error processing {dirname}: {e}")

    return pd.DataFrame(results)

def create_heatmaps(df, output_dir):
    """Create heatmaps for different hyperparameter combinations."""
    os.makedirs(output_dir, exist_ok=True)
    
    vmin = 1e-3
    vmax = 1e-2
    cmap = 'YlOrRd'

    # Rename columns for display
    df = df.rename(columns={'L': 'Layers', 'B': 'Blocks'})
    
    # Create heatmaps for different M values
    for m in df['M'].unique():
        df_m = df[df['M'] == m]
        
        # Heatmap for Layers vs Blocks with HP=8 (train and test side by side)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
        
        # Train error heatmap
        df_hp8 = df_m[df_m['HP'] == 8]
        pivot_train = df_hp8.pivot_table(
            values='train_rel_error',
            index='Layers',
            columns='Blocks',
            aggfunc='mean'
        )
        sns.heatmap(pivot_train, annot=True, fmt='.4e', cmap=cmap, ax=ax1, 
                   norm=LogNorm(vmin=vmin, vmax=vmax))
        ax1.set_title(f'Train Relative Error (M={m}, HP=8)')
        ax1.set_xlabel('Number of blocks')
        ax1.set_ylabel('Number of layers')

        # Test error heatmap
        pivot_test = df_hp8.pivot_table(
            values='test_rel_error',
            index='Layers',
            columns='Blocks',
            aggfunc='mean'
        )
        sns.heatmap(pivot_test, annot=True, fmt='.4e', cmap=cmap, ax=ax2,
                   norm=LogNorm(vmin=vmin, vmax=vmax))
        ax2.set_title(f'Test Relative Error (M={m}, HP=8)')
        ax2.set_xlabel('Number of blocks')
        ax2.set_ylabel('Number of layers')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'heatmap_L_B_M{m}_HP8.png'))
        plt.close()
        
        # Heatmap for Layers vs HP with Blocks=8 (train and test side by side)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
        
        # Train error heatmap
        df_b8 = df_m[df_m['Blocks'] == 8]
        pivot_train = df_b8.pivot_table(
            values='train_rel_error',
            index='Layers',
            columns='HP',
            aggfunc='mean'
        )
        sns.heatmap(pivot_train, annot=True, fmt='.4e', cmap=cmap, ax=ax1,
                   norm=LogNorm(vmin=vmin, vmax=vmax))
        ax1.set_title(f'Train Relative Error (M={m}, Blocks=8)')
        ax1.set_xlabel('Number of projection heads')
        ax1.set_ylabel('Number of layers')
        
        # Test error heatmap
        pivot_test = df_b8.pivot_table(
            values='test_rel_error',
            index='Layers',
            columns='HP',
            aggfunc='mean'
        )
        sns.heatmap(pivot_test, annot=True, fmt='.4e', cmap=cmap, ax=ax2,
                   norm=LogNorm(vmin=vmin, vmax=vmax))
        ax2.set_title(f'Test Relative Error (M={m}, Blocks=8)')
        ax2.set_xlabel('Number of projection heads')
        ax2.set_ylabel('Number of layers')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'heatmap_L_HP_M{m}_B8.png'))
        plt.close()

def create_line_plots(df, output_dir):
    """Create line plots showing the effect of different hyperparameters."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Effect of number of layers (L) - train and test side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # Train error
    for b in sorted(df['B'].unique()):
        df_b = df[df['B'] == b]
        ax1.semilogy(df_b['L'], df_b['train_rel_error'], 
                marker='o', label=f'B={b}')
    ax1.set_xlabel('Number of Layers (L)')
    ax1.set_ylabel('Train Relative Error (log scale)')
    ax1.set_title('Effect of Number of Layers (Train)')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.minorticks_on()
    
    # Test error
    for b in sorted(df['B'].unique()):
        df_b = df[df['B'] == b]
        ax2.semilogy(df_b['L'], df_b['test_rel_error'], 
                marker='o', label=f'B={b}')
    ax2.set_xlabel('Number of Layers (L)')
    ax2.set_ylabel('Test Relative Error (log scale)')
    ax2.set_title('Effect of Number of Layers (Test)')
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.minorticks_on()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lineplot_layers.png'))
    plt.close()
    
    # Effect of number of projection heads (HP) - train and test side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # Train error
    for b in sorted(df['B'].unique()):
        df_b = df[df['B'] == b]
        ax1.semilogy(df_b['HP'], df_b['train_rel_error'], 
                marker='o', label=f'B={b}')
    ax1.set_xlabel('Number of Projection Heads (HP)')
    ax1.set_ylabel('Train Relative Error (log scale)')
    ax1.set_title('Effect of Number of Projection Heads (Train)')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.minorticks_on()
    
    # Test error
    for b in sorted(df['B'].unique()):
        df_b = df[df['B'] == b]
        ax2.semilogy(df_b['HP'], df_b['test_rel_error'], 
                marker='o', label=f'B={b}')
    ax2.set_xlabel('Number of Projection Heads (HP)')
    ax2.set_ylabel('Test Relative Error (log scale)')
    ax2.set_title('Effect of Number of Projection Heads (Test)')
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.minorticks_on()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lineplot_heads.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter results from a specific subdirectory')
    parser.add_argument('subdir', type=str, help='Subdirectory name in out/bench/results/ to analyze')
    args = parser.parse_args()
    
    base_dir = os.path.join('out/bench/results', args.subdir)
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        return
        
    # Create output directory based on input subdirectory
    output_base = os.path.join('out/bench/results', f'{args.subdir}_analysis')
    os.makedirs(output_base, exist_ok=True)
    # Remove existing CSV and PNG files in output directory
    for f in os.listdir(output_base):
        if f.endswith('.csv') or f.endswith('.png'):
            os.remove(os.path.join(output_base, f))

    df = collect_results(base_dir)

    # Create visualizations
    create_heatmaps(df, output_base)
    create_line_plots(df, output_base)
    
    # Save the processed data
    csv_path = os.path.join(output_base, 'hyperparameter_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Plots, CSV have been saved to '{output_base}'")

if __name__ == '__main__':
    main() 