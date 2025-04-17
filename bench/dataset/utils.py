import os
import copy
import numpy as np
import torch
from torch.utils.data import TensorDataset, Subset

import bench
from bench.dataset.timeseries import TimeseriesDataset, TimeseriesDatasetTransform

#======================================================================#
class TransformTensorDataset(TensorDataset):
    def __init__(self, *tensors, transform=None):
        super().__init__(*tensors)
        self.transform = transform

    def __getitem__(self, index):
        x = super().__getitem__(index)
        return self.transform(x)
    
class NormalizationTransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std
    
#======================================================================#
def load_dataset(
        dataset_name: str,
        DATADIR_BASE: str,
        PROJDIR: str,
        force_reload: bool = False,
        mesh: bool = False,
        cells: bool = False,
        max_cases: int = None,
        max_steps: int = None,
        init_step: int = None,
        init_case: int = None,
        exclude: bool = True,
        train_rollout_noise: float = 0.
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

        transform_kwargs = dict(mesh=mesh, cells=cells, train_rollout_noise=train_rollout_noise)
        
        dataset_kwargs = dict(
            force_reload=force_reload,
            max_cases=max_cases,
            max_steps=max_steps,
            init_step=init_step,
        )

        train_transform = TimeseriesDatasetTransform(dataset_name, **transform_kwargs)
        test_transform  = TimeseriesDatasetTransform(dataset_name, **transform_kwargs)

        train_data = TimeseriesDataset(DATADIR, PROJDIR, 'train', transform=train_transform, **dataset_kwargs, init_case=init_case, exclude=exclude)
        test_data  = TimeseriesDataset(DATADIR, PROJDIR, 'test' , transform=test_transform , **dataset_kwargs, exclude=exclude)

        test_data.transform.apply_normalization_stats(train_data.norm_stats)
        
        # Looks like there is some disparity bw train_data and test_data
        train_data, test_data = split_timeseries_dataset(train_data, split=[0.8, 0.2])
        
        return train_data, test_data
        
    else:
        raise ValueError(f"Dataset {dataset_name} not found.") 

#======================================================================#
def split_timeseries_dataset(dataset, split=None, indices=None):
    if split is None and indices is None:
        raise ValueError('split_timeseries_dataset: pass in either indices or split')

    num_cases = dataset.num_cases
    included_cases = dataset.included_cases

    if indices is None:
        indices = [int(s * num_cases) for s in split]
        indices[-1] += num_cases - sum(indices)
    indices = torch.utils.data.random_split(range(num_cases), indices)

    num_split = len(indices)
    subsets = [copy.deepcopy(dataset) for _ in range(num_split)]

    for s in range(num_split):
        subset = subsets[s]
        subset.included_cases = [included_cases[i] for i in indices[s]]
        subset.num_cases = len(subset.included_cases)
        
    # assert there is no overlap between the included cases
    for split1 in range(num_split):
        for split2 in range(num_split):
            if split1 != split2:
                assert not any(c in subsets[split1].included_cases for c in subsets[split2].included_cases)

    return subsets
    
#======================================================================#
#