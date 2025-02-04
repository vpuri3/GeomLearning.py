#
import torch

import os
import yaml
from tqdm import tqdm
from jsonargparse import ArgumentParser, CLI
from dataclasses import dataclass

# local
import am
import mlutils

#===============#
PROJDIR      = '/home/vedantpu/.julia/dev/GeomLearning.py'
DATADIR_BASE = '/home/shared/'
#===============#

#===============#
# PROJDIR = '/ocean/projects/.../vedantpu/GeomLearning.py'
# DATADIR_BASE = 'data/'
#===============#

DATADIR_RAW        = os.path.join(DATADIR_BASE, 'netfabb_ti64_hires_raw')
DATADIR_TIMESERIES = os.path.join(DATADIR_BASE, 'netfabb_ti64_hires_timeseries')
DATADIR_FINALTIME  = os.path.join(DATADIR_BASE, 'netfabb_ti64_hires_finaltime')

SUBDIRS = [
    r'data_0-100',
    r'data_100-200',
    r'data_200-300',
    r'data_300-400',
    r'data_400-500',
    r'data_500-600',
    r'data_600-1000',
]

CASEDIR = os.path.join('.', 'out', 'am')

#======================================================================#
def train_timeseries(cfg, device):
    DISTRIBUTED = mlutils.is_torchrun()
    LOCAL_RANK = int(os.environ['LOCAL_RANK']) if DISTRIBUTED else 0

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    #=================#
    # DATA
    #=================#

    DATADIR = os.path.join(DATADIR_TIMESERIES, r"data_0-100")

    transform = am.TimeseriesDatasetTransform(
        disp=cfg.disp, vmstr=cfg.vmstr, temp=cfg.temp, mesh=cfg.GNN,
        merge=cfg.merge, interpolate=cfg.interpolate, metadata=False,
    )
    dataset = am.TimeseriesDataset(DATADIR, transform=transform, merge=cfg.merge)
    _data, data_ = am.split_timeseries_dataset(dataset, split=[0.8, 0.2])

    # # run small experiments
    # N1 = 30
    # N2 = 10
    # _data, data_ = am.split_timeseries_dataset(dataset, indices=[range(N1), range(N1,N1+N2)])

    # # smaller still experiments (3 (worst), 5, 6, 8, 9)
    # _data, = am.split_timeseries_dataset(dataset, indices=[[3,5,6,8,9]])
    # data_ = None

    #=================#
    # MODEL
    #=================#

    # TODO: add mask_bulk as input
    # TODO: modify mask_bulk parameters. make interface sharper

    ci = 3 + 2 + cfg.disp + cfg.vmstr + cfg.temp # (x, y, z, t, dt, fields...)
    ce = 3
    co = cfg.disp + cfg.vmstr + cfg.temp

    if cfg.GNN:
        model = am.MeshGraphNet(ci, ce, co, cfg.gnn_width, cfg.gnn_num_layers)
    elif cfg.TRA:
        model = am.Transolver(
            space_dim=ci, out_dim=co, fun_dim=0,
            n_hidden=cfg.tra_width, n_layers=cfg.tra_num_layers,
            n_head=cfg.tra_num_heads, mlp_ratio=cfg.tra_mlp_ratio,
            # slice_num=32,
        )
    else:
        raise NotImplementedError()

    model = am.MaskedModel(model, mask=cfg.mask, mask_bulk=cfg.mask_bulk)

    #=================#
    # TRAIN
    #=================#

    callback = am.TimeseriesCallback(case_dir, cfg.GNN, num_eval_cases=cfg.num_eval_cases, autoreg_start=cfg.autoreg_start)
    lossfun = torch.nn.MSELoss()
    batch_lossfun = am.MaskedLoss(cfg.mask)

    if cfg.train:
        if cfg.epochs > 0:
            # # v100-32 GB
            # _batch_size = 4 if len(_data.case_files) > 2 else 1
            # batch_size_ = _batch_size_ = 12

            # RTX 2070-12 GB
            _batch_size = 1
            batch_size_ = _batch_size_ = 1

            kw = dict(
                device=device, gnn_loader=True, stats_every=cfg.epochs//10,
                Opt='AdamW', weight_decay=cfg.weight_decay, lossfun=lossfun, epochs=cfg.epochs,
                _batch_size=_batch_size, batch_size_=batch_size_, _batch_size_=_batch_size_,
                batch_lossfun=batch_lossfun,
            )

            # kw = dict(lr=5e-4, **kw,)
            kw = dict(lr=1e-3, Schedule="OneCycleLR", **kw,)
            trainer = mlutils.Trainer(model, _data, data_, **kw)
            trainer.add_callback('epoch_end', callback)
            trainer.train()

    #=================#
    # ANALYSIS
    #=================#

    trainer = mlutils.Trainer(model, _data, data_, device=device)
    callback.load(trainer)
    callback(trainer, final=True)

    return

#======================================================================#
def train_finaltime(cfg, device):
    DISTRIBUTED = mlutils.is_torchrun()
    LOCAL_RANK = int(os.environ['LOCAL_RANK']) if DISTRIBUTED else 0

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    #=================#
    # DATA
    #=================#

    datasets = []
    for (idir, DIR) in enumerate(SUBDIRS):
        if idir == 2:
            break
        DATADIR = os.path.join(DATADIR_FINALTIME, DIR)
        dataset = am.FinaltimeDataset(DATADIR)#, force_reload=True)#, transform=transform)
        datasets.append(dataset)
        
    transform = am.FinaltimeDatasetTransform(disp=cfg.disp, vmstr=cfg.vmstr, temp=cfg.temp, mesh=cfg.GNN)
    dataset = am.CompositeDataset(*datasets, transform=transform)
    _data, data_ = torch.utils.data.random_split(dataset, [0.8, 0.2])
    
    # _data = torch.utils.data.random_split(dataset, [0.01, 0.99])[0] # works
    # # _data = torch.utils.data.random_split(dataset, [0.05, 0.95])[0] # fails
    # data_ = None
    # print(len(_data))
    # # assert False

    #=================#
    # MODEL
    #=================#

    ci = 3
    ce = 3
    co = cfg.disp + cfg.vmstr + cfg.temp

    if cfg.GNN:
        model = am.MeshGraphNet(ci, ce, co, cfg.gnn_width, cfg.gnn_num_layers)
    elif cfg.TRA:
        model = am.Transolver(
            space_dim=ci, out_dim=co, fun_dim=0,
            n_hidden=cfg.tra_width, n_layers=cfg.tra_num_layers,
            n_head=cfg.tra_num_heads, mlp_ratio=cfg.tra_mlp_ratio,
            slice_num=64,
        )
    else:
        raise NotImplementedError()

    #=================#
    # TRAIN
    #=================#

    lossfun  = torch.nn.MSELoss()
    callback = am.FinaltimeCallback(case_dir, cfg.GNN, num_eval_cases=cfg.num_eval_cases)

    if cfg.train:
        if cfg.epochs > 0:
            _batch_size  = 1
            batch_size_  = 1
            _batch_size_ = 1

            kw = dict(
                device=device, gnn_loader=True, stats_every=cfg.epochs//10,
                Opt='AdamW', weight_decay=cfg.weight_decay, lossfun=lossfun, epochs=cfg.epochs,
                _batch_size=_batch_size, batch_size_=batch_size_, _batch_size_=_batch_size_
            )

            # kw = dict(lr=5e-4, **kw,)
            kw = dict(lr=1e-3, Schedule="OneCycleLR", **kw,)
            trainer = mlutils.Trainer(model, _data, data_, **kw)
            trainer.add_callback('epoch_end', callback)
            trainer.train()

    #=================#
    # ANALYSIS
    #=================#

    trainer = mlutils.Trainer(model, _data, data_, device=device)
    callback.load(trainer)
    callback(trainer, final=True)

    return

#======================================================================#
def filter_dataset():
    """
    remove cases with
    - extremely large meshes (>500k edges), (100k verts)
    - extremely large displacements
    - large aspect ratio of elements
    - too few time-steps (< 10)
    - extremely thin parts
    - pre-training filteing: train a small model and remove cases with large losses

    Tasks:
    1. create a table of data-statistics.
    2. identify cases that are to be excluded - view/ analyze key features
    3. identify based on case_name
    4. create list of case_names
    5. then create mechanism in TimeseriesDataset/ FinaltimeDataset to exclude
    
    Overall goal: run filtering based on finaltime dataset
    """
    
    DISTRIBUTED = mlutils.is_torchrun()
    LOCAL_RANK = int(os.environ['LOCAL_RANK']) if DISTRIBUTED else 0
    
    if LOCAL_RANK != 0:
        return

    import numpy as np
    import pandas as pd

    import seaborn as sns
    import matplotlib.pyplot as plt
    
    stats = {
        # mesh
        'num_vertices': [],
        'num_edges': [],
        'min_aspect_ratio': [],
        'max_aspect_ratio': [],

        # fields
        'max_z': [],
        'max_disp': [],
        'max_vmstr': [],

        # metadata
        'datadir': [],
        'case_name': [],
        # 'num_time_steps': [],
    }

    for (idir, DIR) in enumerate(SUBDIRS):
        DATADIR = os.path.join(DATADIR_FINALTIME, DIR)
        dataset = am.FinaltimeDataset(DATADIR)
    
        print(DATADIR)
        
        for case in dataset:
            # Extract basic metadata
            stats['datadir'].append(DATADIR)
            stats['case_name'].append(case.metadata['case_name'])
            # stats['num_time_steps'].append(len(case.metadata.time_steps))
            
            # Mesh statistics
            stats['num_vertices'].append(case.pos.size(0))
            stats['num_edges'].append(case.edge_index.size(1))
            
            # aspect_ratios = am.compute_aspect_ratios(case.pos.numpy(), case.elems.numpy())
            # stats['min_aspect_ratio'].append(np.min(aspect_ratios))
            # stats['max_aspect_ratio'].append(np.max(aspect_ratios))

            stats['min_aspect_ratio'].append(-1)
            stats['max_aspect_ratio'].append(-1)

            # fields
            stats['max_z'].append(torch.max(case.pos[:,2]).item())
            stats['max_disp'].append(torch.max(case.disp[:,2]).item())
            stats['max_vmstr'].append(torch.max(case.vmstr).item())

            del case

    # Create DataFrame
    df = pd.DataFrame(stats)
    
    # derived statistics
    df['edges_per_vert'] = df['num_edges'] / df['num_vertices']
    
    # Create output directory based on mode
    case_dir = os.path.join(PROJDIR, 'analysis')
    os.makedirs(case_dir, exist_ok=True)
    
    # Save and display statistics
    stats_file = os.path.join(case_dir, 'dataset_statistics.csv')
    df.to_csv(stats_file, index=False)
    
    print("Dataset statistics:")
    print(df.describe())

    # Create probability density plots
    numerical_cols = df.select_dtypes(include=['number']).columns
    
    # Create plots
    plt.figure(figsize=(15, 10))
    plt.title('Probability density')
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(4, 4, i)
        sns.kdeplot(df[col], fill=True, warn_singular=False)
        plt.title(f'{col}')
        plt.xlabel(col)
        plt.ylabel('Density')

    plt.tight_layout()
    plot_file = os.path.join(case_dir, 'density_plots.png')
    plt.savefig(plot_file)
    plt.close()

    return

#======================================================================#
def vis_finaltime(cfg, num_workers=None):

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    for DIR in SUBDIRS:
        DATADIR = os.path.join(DATADIR_FINALTIME, DIR)
        dataset = am.FinaltimeDataset(DATADIR)
        vis_dir = os.path.join(case_dir, DIR)
        os.makedirs(vis_dir, exist_ok=False)
    
        print(vis_dir)

        for icase in tqdm(range(len(dataset))):
            data = dataset[icase]
            ii = str(icase).zfill(3)
            case_name = data.metadata['case_name']
            out_file = os.path.join(vis_dir, f'{ii}_{case_name}.vtu')
            am.visualize_pyv(data, out_file)

    return

#======================================================================#
def vis_timeseries(cfg, num_workers=12):

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    for (idir, DIR) in enumerate(SUBDIRS):
        DATADIR  = os.path.join(DATADIR_TIMESERIES, DIR)
        dataset  = am.TimeseriesDataset(DATADIR, merge=cfg.merge)
        vis_dir  = os.path.join(case_dir, DIR)
    
        if idir > 1:
            break
        
        print(DIR)

        case_names = [f[:-3] for f in os.listdir(DATADIR) if f.endswith(".pt")]
        num_cases = len(case_names)
    
        for icase in tqdm(range(num_cases)):
            case_name = case_names[icase]
            idx_case  = dataset.case_range(case_name)
            case_data = dataset[idx_case]
            out_dir   = os.path.join(vis_dir, f'case{str(icase).zfill(2)}')
            am.visualize_timeseries_pyv(case_data, out_dir, icase, merge=cfg.merge)

    return

#======================================================================#
def test_extraction():
    ext_dir = "/home/shared/netfabb_ti64_hires_out/extracted/SandBox/"
    out_dir = "/home/shared/netfabb_ti64_hires_out/tmp/"
    errfile = os.path.join(out_dir, "error.txt")

    # consider a single case
    case_dir = os.path.join(ext_dir, "33084_344fec27_2")
    # case_dir = os.path.join(ext_dir, "101635_11b839a3_5")
    # case_dir = os.path.join(ext_dir, "83419_82b6bccd_0")
    # case_dir = os.path.join(ext_dir, "77980_f6ed5970_4")

    info = am.get_case_info(case_dir)
    print(info)
    # results = am.get_timeseries_results(case_dir)

    am.extract_from_dir(ext_dir, out_dir, errfile, timeseries=True)

    return

def do_extraction():

    # zip_file = os.path.join(DATADIR_RAW, "data_600-1000.zip")
    # out_dir  = os.path.join(DATADIR_FINALTIME, "data_600-1000")
    # am.extract_from_zip(zip_file, out_dir)
    # am.extract_from_zip(zip_file, out_dir, timeseries=True)

    return

#======================================================================#
@dataclass
class Config:
    '''
    Train grah neural networks on time series AM data
    '''

    # different modes
    analysis: bool = False
    extraction: bool = False
    visualization: bool = False
    train: bool = False
    eval: bool = False
    timeseries: bool = False

    # case configuration
    exp_name: str = 'exp'
    seed: int = 123

    # fields
    disp: bool  = True
    vmstr: bool = False
    temp: bool  = False

    # model
    GNN: bool = False
    TRA: bool = False

    # GNN
    gnn_width: int = 96
    gnn_num_layers: int = 5

    # TRA
    tra_width: int = 128
    tra_num_layers: int = 5
    tra_num_heads: int = 8
    tra_mlp_ratio: float = 2.0

    # timeseries  dataset
    merge: bool = True
    mask: bool = True
    blend: bool = True
    mask_bulk: bool = False
    interpolate: bool = True

    # training arguments
    epochs: int = 100
    weight_decay: float = 1e-2

    # eval arguments
    num_eval_cases: int = 20
    autoreg_start: int = 10

if __name__ == "__main__":
    
    #===============#
    cfg = CLI(Config, as_positional=False)
    #===============#

    if not (cfg.analysis or cfg.extraction or cfg.visualization or cfg.train or cfg.eval):
        print("No mode selected")
        exit()

    DISTRIBUTED = mlutils.is_torchrun()
    LOCAL_RANK = int(os.environ['LOCAL_RANK']) if DISTRIBUTED else 0
    device = mlutils.select_device()

    #===============#
    mlutils.set_seed(cfg.seed)
    #===============#

    if cfg.analysis:
        filter_dataset()
        exit()

    if cfg.extraction:
        do_extraction()
        exit()

    if cfg.visualization:
        # am.extract_zips(DATADIR_RAW, DATADIR_FINALTIME)
        # vis_finaltime(cfg)

        # test_extraction()
        # am.extract_zips(DATADIR_RAW, DATADIR_TIMESERIES, timeseries=True, num_workers=12)
        # vis_timeseries(cfg)
        pass

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    if cfg.train:
        if os.path.exists(case_dir):
            nd = len([dir for dir in os.listdir(CASEDIR) if dir.startswith(cfg.exp_name)])
            case_dir = case_dir + str(nd).zfill(2)
            cfg.exp_name = cfg.exp_name + str(nd).zfill(2)

        if DISTRIBUTED:
            torch.distributed.barrier()

        if LOCAL_RANK == 0:
            os.makedirs(case_dir)
            config_file = os.path.join(case_dir, 'config.yaml')
            print(f'Saving config to {config_file}')
            with open(config_file, 'w') as f:
                yaml.safe_dump(vars(cfg), f)

    if cfg.eval:
        assert os.path.exists(case_dir)
        config_file = os.path.join(case_dir, 'config.yaml')
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)

    if DISTRIBUTED:
        torch.distributed.barrier()

    if cfg.timeseries:
        train_timeseries(cfg, device)
    else:
        train_finaltime(cfg, device)


    #===============#
    mlutils.dist_finalize()
    #===============#

    exit()
#