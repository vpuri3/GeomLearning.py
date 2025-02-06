#
import os
import torch

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from .finaltime import FinaltimeDataset

__all__ = [
    'save_dataset_statistics',
    'compute_filtered_dataset_statistics',
    'compute_dataset_statistics',
    'make_exclusion_list',
]

#======================================================================#
def save_dataset_statistics(df, case_dir):
    """
    Save and plot dataset statistics
    """
    # Save dataset statistics
    stats_csv = os.path.join(case_dir, 'dataset_statistics.csv')
    stats_txt = os.path.join(case_dir, 'dataset_statistics.txt')
    stats_png = os.path.join(case_dir, 'dataset_statistics.png')

    # save stats.csv
    df.to_csv(stats_csv, index=False)

    # save stats.txt
    with open(stats_txt, 'w') as f:
        f.write(str(df.describe()))

    # print stats
    print(df.describe())

    # Create probability density plots
    numerical_cols = df.select_dtypes(include=['number']).columns
    
    # Create plots
    plt.figure(figsize=(20, 15))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(3, 3, i)
        sns.kdeplot(df[col], fill=True, warn_singular=False)
        plt.title(f'PDF of {col}', pad=10, fontsize=18)
        plt.xlabel(col, labelpad=5)
        plt.ylabel("Density")
        plt.yticks([])
        plt.tight_layout(pad=3.0)

    plt.tight_layout()
    plot_file = os.path.join(case_dir, stats_png)
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    
    return

#======================================================================#
def compute_filtered_dataset_statistics(PROJDIR):
    """
    Compute statistics on the filtered dataset (excluding problematic cases)
    """
    # Load full statistics
    stats_csv = os.path.join(PROJDIR, 'analysis', 'dataset_statistics.csv')
    df = pd.read_csv(stats_csv)
    
    # Load exclusion list
    exclusion_list_file = os.path.join(PROJDIR, 'analysis', 'exclusion_list.txt')
    with open(exclusion_list_file, 'r') as f:
        exclusion_list = [line.strip() for line in f.readlines()]
    
    # Filter dataset
    filtered_df = df[~df['case_name'].isin(exclusion_list)]
    
    # Save filtered statistics
    filtered_case_dir = os.path.join(PROJDIR, 'analysis', 'filtered')
    os.makedirs(filtered_case_dir, exist_ok=True)
    save_dataset_statistics(filtered_df, filtered_case_dir)
    
    return filtered_df

#======================================================================#
def compute_dataset_statistics(PROJDIR, DATADIR_FINALTIME, SUBDIRS):
    """
    Compute statistics on the full dataset
    """
    # Create output directory based on mode
    case_dir = os.path.join(PROJDIR, 'analysis')
    os.makedirs(case_dir, exist_ok=True)
    
    # Create directory for aspect ratios cache
    aspect_ratios_dir = os.path.join(case_dir, 'mesh_aspect_ratios')
    os.makedirs(aspect_ratios_dir, exist_ok=True)

    # Save dataset statistics
    stats_csv = os.path.join(case_dir, 'dataset_statistics.csv')
    stats_txt = os.path.join(case_dir, 'dataset_statistics.txt')
    stats_png = os.path.join(case_dir, 'dataset_statistics.png')

    stats = {
        # mesh
        'num_vertices': [],
        'num_edges': [],
        'avg_aspect_ratio': [],
        'max_aspect_ratio': [],

        # fields
        'max_z': [],
        'max_disp': [],
        'max_vmstr': [],

        # metadata
        'datadir': [],
        'case_name': [],
    }

    for (idir, DIR) in enumerate(SUBDIRS):
        DATADIR = os.path.join(DATADIR_FINALTIME, DIR)
        dataset = FinaltimeDataset(DATADIR)
    
        # subdirectory for cached aspect ratios
        aspect_ratio_subdir = os.path.join(aspect_ratios_dir, DIR)
        os.makedirs(aspect_ratio_subdir, exist_ok=True)
        
        if not os.path.exists(aspect_ratio_subdir):
            print(f"Computing aspect ratios for {DIR}")
            
        for case in dataset:
            # Extract basic metadata
            stats['datadir'].append(DATADIR)
            stats['case_name'].append(case.metadata['case_name'])
            
            # Mesh statistics
            stats['num_vertices'].append(case.pos.size(0))
            stats['num_edges'].append(case.edge_index.size(1))
            
            # Cache aspect ratios
            aspect_ratios_file = os.path.join(aspect_ratio_subdir, case.metadata['case_name'] + '.csv')
            
            if os.path.exists(aspect_ratios_file):
                aspect_ratios = np.loadtxt(aspect_ratios_file)
            else:
                pos, elems = case.pos.numpy(), case.elems.numpy()
                aspect_ratios = am.compute_aspect_ratios(pos, elems)
                np.savetxt(aspect_ratios_file, aspect_ratios)
                del pos, elems

            stats['avg_aspect_ratio'].append(np.mean(aspect_ratios))
            stats['max_aspect_ratio'].append(np.max(aspect_ratios))

            # fields
            stats['max_z'].append(torch.max(case.pos[:,2]).item())
            stats['max_disp'].append(torch.max(case.disp[:,2]).item())
            stats['max_vmstr'].append(torch.max(case.vmstr).item())

            del case

    # Create DataFrame
    df = pd.DataFrame(stats)

    # derived statistics
    df['edges_per_vert'] = df['num_edges'] / df['num_vertices']

    # Save statistics
    save_dataset_statistics(df, case_dir)

    return df

#======================================================================#
def make_exclusion_list(PROJDIR):
    """
    make a list of case_names to exclude based on statistics
    """
    
    # load stats.csv
    stats_csv = os.path.join(PROJDIR, 'analysis', 'dataset_statistics.csv')
    df = pd.read_csv(stats_csv)

    exclusion_list = []
    # exclusion_list += df[df['num_vertices'] > 1e5]['case_name'].tolist() # too many verts
    exclusion_list += df[df['num_edges'] > 5e5]['case_name'].tolist() # too many edges
    # exclusion_list += df[df['avg_aspect_ratio'] > 10]['case_name'].tolist() # thin features
    exclusion_list += df[df['max_aspect_ratio'] > 20]['case_name'].tolist() # thin features
    exclusion_list += df[df['max_z'] < 30]['case_name'].tolist() # too short
    exclusion_list += df[df['max_disp'] > 2]['case_name'].tolist() # bad displacement
    exclusion_list += df[df['max_vmstr'] > 4000]['case_name'].tolist() # bad stress
    exclusion_list += df[df['edges_per_vert'] < 5]['case_name'].tolist() # thin features

    # save exclusion_list.txt
    exclusion_list_file = os.path.join(PROJDIR, 'analysis', 'exclusion_list.txt')

    print(f"Saving exclusion list to {exclusion_list_file} with {len(exclusion_list)} / {len(df)} cases.")

    with open(exclusion_list_file, 'w') as f:
        for case_name in exclusion_list:
            f.write(f"{case_name}\n")

    return exclusion_list

#======================================================================#
#