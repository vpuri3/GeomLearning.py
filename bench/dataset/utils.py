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
        import scipy.io as scio

        DATADIR = os.path.join(DATADIR_BASE, 'Geo-FNO', 'plasticity')
        data_path = os.path.join(DATADIR, 'plas_N987_T20.mat')
        
        N = 987
        ntrain = 900
        ntest = 80

        s1 = 101
        s2 = 31
        T = 20
        Deformation = 4

        r1 = 1
        r2 = 1
        s1 = int(((s1 - 1) / r1) + 1)
        s2 = int(((s2 - 1) / r2) + 1)

        data = scio.loadmat(data_path)
        input = torch.tensor(data['input'], dtype=torch.float)
        output = torch.tensor(data['output'], dtype=torch.float).transpose(-2, -1)
        x_train = input[:ntrain, ::r1][:, :s1].reshape(ntrain, s1, 1).repeat(1, 1, s2)
        x_train = x_train.reshape(ntrain, -1, 1)
        y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
        y_train = y_train.reshape(ntrain, -1, Deformation, T)
        x_test = input[-ntest:, ::r1][:, :s1].reshape(ntest, s1, 1).repeat(1, 1, s2)
        x_test = x_test.reshape(ntest, -1, 1)
        y_test = output[-ntest:, ::r1, ::r2][:, :s1, :s2]
        y_test = y_test.reshape(ntest, -1, Deformation, T)

        x_normalizer = bench.UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)

        x = np.linspace(0, 1, s1)
        y = np.linspace(0, 1, s2)
        x, y = np.meshgrid(x, y)
        pos = np.c_[x.ravel(), y.ravel()]
        pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)

        pos_train = pos.repeat(ntrain, 1, 1)
        pos_test = pos.repeat(ntest, 1, 1)

        t = np.linspace(0, 1, T)
        t = torch.tensor(t, dtype=torch.float).unsqueeze(0)
        t_train = t.repeat(ntrain, 1)
        t_test = t.repeat(ntest, 1)
        
        print(pos_train.shape, t_train.shape, x_train.shape, y_train.shape)
        exit()

        train_data = TensorDataset(pos_train, t_train, x_train, y_train)
        test_data  = TensorDataset(pos_test, t_test, x_test, y_test)
        
        # collate_fn = random_collate_fn
        
        return train_data, test_data, x_normalizer, y_normalizer
        
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
        import scipy.io as scio

        DATADIR = os.path.join(DATADIR_BASE, 'FNO', 'darcy')

        train_path = os.path.join(DATADIR, 'piececonst_r421_N1024_smooth1.mat')
        test_path = os.path.join(DATADIR, 'piececonst_r421_N1024_smooth2.mat')
        ntrain = 1000
        ntest = 200
        
        r = 5 # downsample
        h = int(((421 - 1) / r) + 1)
        s = h
        dx = 1.0 / s

        train_data = scio.loadmat(train_path)
        x_train = train_data['coeff'][:ntrain, ::r, ::r][:, :s, :s]
        x_train = x_train.reshape(ntrain, -1, 1)
        x_train = torch.from_numpy(x_train).float()
        y_train = train_data['sol'][:ntrain, ::r, ::r][:, :s, :s]
        y_train = y_train.reshape(ntrain, -1, 1)
        y_train = torch.from_numpy(y_train)

        test_data = scio.loadmat(test_path)
        x_test = test_data['coeff'][:ntest, ::r, ::r][:, :s, :s]
        x_test = x_test.reshape(ntest, -1, 1)
        x_test = torch.from_numpy(x_test).float()
        y_test = test_data['sol'][:ntest, ::r, ::r][:, :s, :s]
        y_test = y_test.reshape(ntest, -1, 1)
        y_test = torch.from_numpy(y_test)

        x_normalizer = bench.UnitGaussianNormalizer(x_train)
        y_normalizer = bench.UnitGaussianNormalizer(y_train)

        x_train = x_normalizer.encode(x_train)
        y_train = y_normalizer.encode(y_train)

        x_test = x_normalizer.encode(x_test)
        y_test = y_normalizer.encode(y_test)

        x = np.linspace(0, 1, s)
        y = np.linspace(0, 1, s)
        x, y = np.meshgrid(x, y)
        pos = np.c_[x.ravel(), y.ravel()]
        pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)

        pos_train = pos.repeat(ntrain, 1, 1)
        pos_test = pos.repeat(ntest, 1, 1)
        
        input_train = torch.cat([pos_train, x_train], dim=-1)
        output_train = y_train.to(torch.float)
        
        input_test = torch.cat([pos_test, x_test], dim=-1)
        output_test = y_test.to(torch.float)
        
        train_data = TensorDataset(input_train, output_train)
        test_data  = TensorDataset(input_test , output_test )

        return train_data, test_data, x_normalizer, y_normalizer
        
    elif dataset_name == 'navier_stokes':
        import scipy.io as scio

        DATADIR = os.path.join(DATADIR_BASE, 'FNO', 'navier_stokes')
        data_path = os.path.join(DATADIR, 'NavierStokes_V1e-5_N1200_T20.mat')
        
        r = 1
        h = int(((64 - 1) / r) + 1)
        ntrain = 1000
        ntest = 200
        T_in = 10
        T = 10

        data = scio.loadmat(data_path)
        train_a = data['u'][:ntrain, ::r, ::r, :T_in][:, :h, :h, :]
        train_a = train_a.reshape(train_a.shape[0], -1, train_a.shape[-1])
        train_a = torch.from_numpy(train_a)
        train_u = data['u'][:ntrain, ::r, ::r, T_in:T + T_in][:, :h, :h, :]
        train_u = train_u.reshape(train_u.shape[0], -1, train_u.shape[-1])
        train_u = torch.from_numpy(train_u)

        test_a = data['u'][-ntest:, ::r, ::r, :T_in][:, :h, :h, :]
        test_a = test_a.reshape(test_a.shape[0], -1, test_a.shape[-1])
        test_a = torch.from_numpy(test_a)
        test_u = data['u'][-ntest:, ::r, ::r, T_in:T + T_in][:, :h, :h, :]
        test_u = test_u.reshape(test_u.shape[0], -1, test_u.shape[-1])
        test_u = torch.from_numpy(test_u)

        x = np.linspace(0, 1, h)
        y = np.linspace(0, 1, h)
        x, y = np.meshgrid(x, y)
        pos = np.c_[x.ravel(), y.ravel()]
        pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
        pos_train = pos.repeat(ntrain, 1, 1)
        pos_test = pos.repeat(ntest, 1, 1)
        
        # print(pos_train.shape) # [1000, 4096,  2]
        # print(train_a.shape)   # [1000, 4096, 10]
        # print(train_u.shape)   # [1000, 4096, 10]
        
        train_input = torch.cat([pos_train, train_a], dim=-1)
        test_input  = torch.cat([pos_test , test_a ], dim=-1)

        train_output = train_u
        test_output  = test_u
        
        train_dataset = TensorDataset(train_input, train_output)
        test_dataset  = TensorDataset(test_input , test_output )
        
        x_normalizer = bench.IdentityNormalizer()
        y_normalizer = bench.IdentityNormalizer()
        
        # no normalization?

        return train_dataset, test_dataset, x_normalizer, y_normalizer
        
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