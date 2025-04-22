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
        tuple: (train_data, test_data, y_normalizer) containing the loaded datasets and optional normalizer object
    """
    #----------------------------------------------------------------#
    # Geo-FNO datasets
    #----------------------------------------------------------------#
    if dataset_name == 'elasticity':
        DATADIR = os.path.join(DATADIR_BASE, 'Geo-FNO', 'elasticity')
        PATH_Sigma = os.path.join(DATADIR, 'Meshes', 'Random_UnitCell_sigma_10.npy')
        PATH_XY = os.path.join(DATADIR, 'Meshes', 'Random_UnitCell_XY_10.npy')

        input_s = np.load(PATH_Sigma)
        input_s = torch.tensor(input_s, dtype=torch.float).permute(1, 0).unsqueeze(-1)
        input_xy = np.load(PATH_XY)
        input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2, 0, 1)
        
        ntrain = 1000
        ntest = 200
        
        y_normalizer = bench.UnitGaussianNormalizer(input_s[:ntrain])
        input_s = y_normalizer.encode(input_s)

        dataset = TensorDataset(input_xy, input_s)
        train_data = Subset(dataset, range(ntrain))
        test_data = Subset(dataset, range(len(dataset)-ntest, len(dataset)))
        
        return train_data, test_data, bench.IdentityNormalizer(), y_normalizer
    
    elif dataset_name == 'plasticity':
        DATADIR = os.path.join(DATADIR_BASE, 'Geo-FNO', 'plasticity')
        raise NotImplementedError(f'Dataset {dataset_name} not implemented')
        
    elif dataset_name == 'pipe':
        DATADIR = os.path.join(DATADIR_BASE, 'Geo-FNO', 'pipe')
        
        INPUT_X = os.path.join(DATADIR, 'Pipe_X.npy')
        INPUT_Y = os.path.join(DATADIR, 'Pipe_Y.npy')
        OUTPUT_Sigma = os.path.join(DATADIR, 'Pipe_Q.npy')

        ntrain = 1000
        ntest = 200
        N = 1200
        
        r1 = 1
        r2 = 1
        s1 = int(((129 - 1) / r1) + 1)
        s2 = int(((129 - 1) / r2) + 1)

        inputX = np.load(INPUT_X)
        inputX = torch.tensor(inputX, dtype=torch.float)
        inputY = np.load(INPUT_Y)
        inputY = torch.tensor(inputY, dtype=torch.float)
        input = torch.stack([inputX, inputY], dim=-1)

        output = np.load(OUTPUT_Sigma)[:, 0]
        output = torch.tensor(output, dtype=torch.float)
        x_train = input[:N][:ntrain, ::r1, ::r2][:, :s1, :s2]
        y_train = output[:N][:ntrain, ::r1, ::r2][:, :s1, :s2]
        x_test = input[:N][-ntest:, ::r1, ::r2][:, :s1, :s2]
        y_test = output[:N][-ntest:, ::r1, ::r2][:, :s1, :s2]

        x_train = x_train.reshape(ntrain, -1, 2)
        y_train = y_train.reshape(ntrain, -1, 1)

        x_test = x_test.reshape(ntest, -1, 2)
        y_test = y_test.reshape(ntest, -1, 1)

        x_normalizer = bench.UnitGaussianNormalizer(x_train)
        y_normalizer = bench.UnitGaussianNormalizer(y_train)

        x_train = x_normalizer.encode(x_train)
        y_train = y_normalizer.encode(y_train)

        x_test  = x_normalizer.encode(x_test)
        y_test  = y_normalizer.encode(y_test)

        train_data = TensorDataset(x_train, y_train)
        test_data  = TensorDataset(x_test , y_test )

        return train_data, test_data, x_normalizer, y_normalizer
        
    elif dataset_name == 'airfoil_steady':
        DATADIR = os.path.join(DATADIR_BASE, 'Geo-FNO', 'airfoil', 'naca')

        INPUT_X = os.path.join(DATADIR, 'NACA_Cylinder_X.npy')
        INPUT_Y = os.path.join(DATADIR, 'NACA_Cylinder_Y.npy')
        OUTPUT_Sigma = os.path.join(DATADIR, 'NACA_Cylinder_Q.npy')

        ntrain = 1000
        ntest = 200

        r1 = 1
        r2 = 1
        s1 = int(((221 - 1) / r1) + 1)
        s2 = int(((51 - 1) / r2) + 1)

        inputX = np.load(INPUT_X)
        inputX = torch.tensor(inputX, dtype=torch.float)
        inputY = np.load(INPUT_Y)
        inputY = torch.tensor(inputY, dtype=torch.float)
        input = torch.stack([inputX, inputY], dim=-1)

        output = np.load(OUTPUT_Sigma)[:, 4]
        output = torch.tensor(output, dtype=torch.float)

        x_train = input[:ntrain, ::r1, ::r2][:, :s1, :s2]
        y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
        x_test = input[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2]
        y_test = output[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2]

        x_train = x_train.reshape(ntrain, -1, 2)
        y_train = y_train.reshape(ntrain, -1, 1)

        x_test = x_test.reshape(ntest, -1, 2)
        y_test = y_test.reshape(ntest, -1, 1)
        
        x_normalizer = bench.IdentityNormalizer()
        y_normalizer = bench.IdentityNormalizer()
        
        # x_normalizer = bench.UnitCubeNormalizer(x_train)
        # x_train = x_normalizer.encode(x_train)
        # x_test  = x_normalizer.encode(x_test)

        y_normalizer = bench.UnitGaussianNormalizer(y_train)
        y_train = y_normalizer.encode(y_train)
        y_test  = y_normalizer.encode(y_test)
        
        # # input grid extrema
        # x_min = x_train[:,:,0].min()
        # x_max = x_train[:,:,0].max()
        # y_min = x_train[:,:,1].min()
        # y_max = x_train[:,:,1].max()
        # print(f"Grid min/max: {x_min}, {x_max}, {y_min}, {y_max}")
        # # input grid mean, std
        # x_mean = x_train[:,:,0].mean()
        # y_mean = x_train[:,:,1].mean()
        # x_std = x_train[:,:,0].std()
        # y_std = x_train[:,:,1].std()
        # print(f"Grid mean, std: {x_mean}, {x_std}, {y_mean}, {y_std}")

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.scatter(x_train[0,:,0], x_train[0,:,1], c=y_train[0,:,0], cmap='viridis', s=1)
        # plt.xlim(-1, 1)
        # plt.ylim(-1, 1)
        # plt.savefig('airfoil_train.png')
        # exit()

        # # output mean, std
        # o_mean = y_train.mean()
        # o_std = y_train.std()
        # print(f"Output mean, std: {o_mean}, {o_std}")

        train_data = TensorDataset(x_train, y_train)
        test_data  = TensorDataset(x_test , y_test)

        return train_data, test_data, x_normalizer, y_normalizer
        
    #----------------------------------------------------------------#
    # FNO datasets
    #----------------------------------------------------------------#
    elif dataset_name == 'darcy':
        import scipy

        DATADIR = os.path.join(DATADIR_BASE, 'FNO', 'darcy')
        
        train_path = os.path.join(DATADIR, 'piececonst_r421_N1024_smooth1.mat')
        test_path  = os.path.join(DATADIR, 'piececonst_r421_N1024_smooth2.mat')

        train_data = scipy.io.loadmat(train_path)['coeffs']
        test_data = scipy.io.loadmat(test_path)['coeffs']

        train_data = torch.tensor(train_data, dtype=torch.float)
        test_data = torch.tensor(test_data, dtype=torch.float)

        raise NotImplementedError(f'Dataset {dataset_name} not implemented')
        
    elif dataset_name == 'navier_stokes':
        DATADIR = os.path.join(DATADIR_BASE, 'FNO', 'ns')
        raise NotImplementedError(f'Dataset {dataset_name} not implemented')
        
    #----------------------------------------------------------------#
    # MeshGraphNets datasets
    #----------------------------------------------------------------#
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
        
        return train_data, test_data, None, None
        
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