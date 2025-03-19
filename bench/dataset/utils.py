import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, Subset

import bench
from bench.dataset.timeseries import TimeseriesDataset, TimeseriesDatasetTransform

#======================================================================#
def load_dataset(
        dataset_name: str,
        DATADIR_BASE: str,
        force_reload: bool = False,
        mesh: bool = False,
        cells: bool = False,
        max_cases: int = None,
    ):
    """Load a dataset by name.
    
    Args:
        dataset_name (str): Name of the dataset to load ('elasticity', 'airfoil', or 'cylinder_flow')
        
    Returns:
        tuple: (train_data, test_data) containing the loaded datasets
    """
    if dataset_name == 'elasticity':
        DATADIR = os.path.join(DATADIR_BASE, 'GeoFNO', 'elasticity')
        PATH_Sigma = os.path.join(DATADIR, 'Meshes', 'Random_UnitCell_sigma_10.npy')
        PATH_XY = os.path.join(DATADIR, 'Meshes', 'Random_UnitCell_XY_10.npy')

        input_s = np.load(PATH_Sigma)
        input_s = torch.tensor(input_s, dtype=torch.float).permute(1, 0).unsqueeze(-1)
        input_xy = np.load(PATH_XY)
        input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2, 0, 1)
        
        ntrain = 1000
        ntest = 200
        
        y_normalizer = bench.UnitTransformer(input_s[:ntrain])
        input_s = y_normalizer.encode(input_s)

        dataset = TensorDataset(input_xy, input_s)
        train_data = Subset(dataset, range(ntrain))
        test_data = Subset(dataset, range(len(dataset)-ntest, len(dataset)))
        
        return train_data, test_data
        
    elif dataset_name in ['airfoil', 'cylinder_flow']:
        DATADIR = os.path.join(DATADIR_BASE, 'MeshGraphNets', dataset_name)

        train_transform = TimeseriesDatasetTransform(mesh=mesh, cells=cells, dens=dataset_name == 'airfoil')
        test_transform  = TimeseriesDatasetTransform(mesh=mesh, cells=cells, dens=dataset_name == 'airfoil')

        train_data = TimeseriesDataset(DATADIR, 'train', force_reload=force_reload, transform=train_transform, max_cases=max_cases)
        test_data  = TimeseriesDataset(DATADIR, 'test', force_reload=force_reload, transform=test_transform, max_cases=max_cases)
        
        return train_data, test_data
        
    else:
        raise ValueError(f"Dataset {dataset_name} not found.") 
#======================================================================#