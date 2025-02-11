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
from am.dataset.filtering import save_dataset_statistics, compute_filtered_dataset_statistics, compute_dataset_statistics

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
    r'data_1000-1500',
    r'data_1500-2000',
    r'data_2000-2500',
    r'data_2500-3000',
    r'data_3000-3500',
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

    # read in exclusion list
    exclude_list = os.path.join(PROJDIR, 'analysis', 'exclusion_list.txt')
    exclude_list = [line.strip() for line in open(exclude_list, 'r').readlines()]
    
    transform = am.TimeseriesDatasetTransform(
        disp=cfg.disp, vmstr=cfg.vmstr, temp=cfg.temp, mesh=cfg.GNN,
        merge=cfg.merge, interpolate=cfg.interpolate, metadata=False,
    )
    DATADIRS = [os.path.join(DATADIR_TIMESERIES, DIR) for DIR in SUBDIRS]
    DATADIRS = DATADIRS[:5]
    dataset = am.TimeseriesDataset(DATADIRS, merge=cfg.merge, exclude_list=exclude_list, transform=transform, verbose=LOCAL_RANK==0)
    _data, data_ = am.split_timeseries_dataset(dataset, split=[0.8, 0.2])
    
    # # run small experiments
    # N1, N2 = 30, 10
    # _data, data_ = am.split_timeseries_dataset(dataset, indices=[range(N1), range(N1,N1+N2)])

    if LOCAL_RANK == 0:
        print(f"Loaded {len(dataset.case_files)} cases from {DATADIR_TIMESERIES}")
        print(f"Split into {len(_data.case_files)} train and {len(data_.case_files)} test cases")
    
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
            slice_num=64,
        )
    else:
        print(f"No model selected. Choose between GNN or TRA.")
        raise NotImplementedError()

    model = am.MaskedModel(model, mask=cfg.mask, mask_bulk=cfg.mask_bulk)

    #=================#
    # TRAIN
    #=================#

    lossfun = torch.nn.MSELoss()
    batch_lossfun = am.MaskedLoss(cfg.mask)
    callback = am.TimeseriesCallback(case_dir, mesh=cfg.GNN, num_eval_cases=cfg.num_eval_cases, autoreg_start=cfg.autoreg_start)

    if cfg.train and cfg.epochs > 0:
        # # v100-32 GB
        # _batch_size = 4 if len(_data.case_files) > 2 else 1
        # batch_size_ = _batch_size_ = 12

        # RTX 2070-12 GB
        _batch_size = 1
        batch_size_ = _batch_size_ = 1

        kw = dict(
            device=device, gnn_loader=True, stats_every=cfg.epochs//5,
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

    # read in exclusion list
    exclude_list = os.path.join(PROJDIR, 'analysis', 'exclusion_list.txt')
    exclude_list = [line.strip() for line in open(exclude_list, 'r').readlines()]

    datasets = []
    for (idir, DIR) in enumerate(SUBDIRS):
        DATADIR = os.path.join(DATADIR_FINALTIME, DIR)
        dataset = am.FinaltimeDataset(DATADIR, exclude_list=exclude_list)#, force_reload=True)
        datasets.append(dataset)
        
    transform = am.FinaltimeDatasetTransform(disp=cfg.disp, vmstr=cfg.vmstr, temp=cfg.temp, mesh=cfg.GNN)
    dataset = am.CompositeDataset(*datasets, transform=transform)
    _data, data_ = torch.utils.data.random_split(dataset, [0.8, 0.2])
    
    if LOCAL_RANK == 0:
        print(f"Loaded {len(dataset)} cases from {DATADIR_FINALTIME}")
        print(f"Split into {len(_data)} train and {len(data_)} test cases")
    
    #=================#
    # MODEL
    #=================#

    ci = 3 #+ cfg.sdf
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
        print(f"No model selected. Choose between GNN or TRA.")
        raise NotImplementedError()

    #=================#
    # TRAIN
    #=================#

    lossfun  = torch.nn.MSELoss()
    callback = am.FinaltimeCallback(case_dir, mesh=cfg.GNN, num_eval_cases=cfg.num_eval_cases)

    if cfg.train and cfg.epochs > 0:
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
def vis_finaltime(cfg, force_reload=True, max_cases=50, num_workers=None):

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    # read in exclusion list
    exclude_list = os.path.join(PROJDIR, 'analysis', 'exclusion_list.txt')
    exclude_list = [line.strip() for line in open(exclude_list, 'r').readlines()]
    
    # transform = am.FinaltimeDatasetTransform(
    #     disp=cfg.disp, vmstr=cfg.vmstr, temp=cfg.temp,
    #     sdf=cfg.sdf, mesh=True,
    # )

    # SUBDIRS = SUBDIRS[:1]

    for DIR in SUBDIRS:
        DATADIR = os.path.join(DATADIR_FINALTIME, DIR)
        dataset = am.FinaltimeDataset(
            DATADIR,
            exclude_list=exclude_list,
            force_reload=force_reload,
            num_workers=num_workers,
        )
        vis_dir = os.path.join(case_dir, DIR)
        os.makedirs(vis_dir, exist_ok=False)

        num_cases = min(len(dataset), max_cases)

        for icase in tqdm(range(num_cases)):
            data = dataset[icase]
            ii = str(icase).zfill(3)
            case_name = data.metadata['case_name']
            out_file = os.path.join(vis_dir, f'{ii}_{case_name}.vtu')
            am.visualize_pyv(data, out_file)

    return

#======================================================================#
def vis_timeseries(cfg, max_cases=10):

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    for (idir, DIR) in enumerate(SUBDIRS):
        DATADIR  = os.path.join(DATADIR_TIMESERIES, DIR)
        dataset  = am.TimeseriesDataset(DATADIR, merge=cfg.merge)
        vis_dir  = os.path.join(case_dir, DIR)
    
        case_names = [f[:-3] for f in os.listdir(DATADIR) if f.endswith(".pt")]
        num_cases = len(case_names)
    
        print(vis_dir)
        num_cases = min(num_cases, max_cases)

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

    zip_file = os.path.join(DATADIR_RAW, "data_600-1000.zip")
    out_dir  = os.path.join(DATADIR_FINALTIME, "data_600-1000")
    am.extract_from_zip(zip_file, out_dir, timeseries=True)

    # zip_file = os.path.join(DATADIR_RAW, "data_1000-1500.zip")
    # out_dir  = os.path.join(DATADIR_FINALTIME, "data_1000-1500")
    # am.extract_from_zip(zip_file, out_dir, timeseries=True)

    # zip_file = os.path.join(DATADIR_RAW, "data_1500-2000.zip")
    # out_dir  = os.path.join(DATADIR_FINALTIME, "data_1500-2000")
    # am.extract_from_zip(zip_file, out_dir, timeseries=True)

    # zip_file = os.path.join(DATADIR_RAW, "data_2000-2500.zip")
    # out_dir  = os.path.join(DATADIR_FINALTIME, "data_2000-2500")
    # am.extract_from_zip(zip_file, out_dir, timeseries=True)

    # zip_file = os.path.join(DATADIR_RAW, "data_2500-3000.zip")
    # out_dir  = os.path.join(DATADIR_FINALTIME, "data_2500-3000")
    # am.extract_from_zip(zip_file, out_dir, timeseries=True)

    # zip_file = os.path.join(DATADIR_RAW, "data_3000-3500.zip")
    # out_dir  = os.path.join(DATADIR_FINALTIME, "data_3000-3500")
    # am.extract_from_zip(zip_file, out_dir, timeseries=True)

    # zip_file = os.path.join(DATADIR_RAW, "data_0-100.zip")
    # out_dir  = os.path.join(DATADIR_FINALTIME, "dump")
    # am.extract_from_zip(zip_file, out_dir)
    # # am.extract_from_zip(zip_file, out_dir, timeseries=True)

    return

#======================================================================#
@dataclass
class Config:
    '''
    Train neural networks on time series AM data
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
    
    # features
    sdf: bool = False

    # model
    GNN: bool = False
    TRA: bool = False

    # GNN
    gnn_width: int = 128
    gnn_num_layers: int = 5

    # TRA
    tra_width: int = 128
    tra_num_layers: int = 5
    tra_num_heads: int = 8
    tra_mlp_ratio: float = 2.0

    # training arguments
    epochs: int = 100
    weight_decay: float = 1e-3

    # timeseries  dataset
    merge: bool = True
    mask: bool = True
    blend: bool = False
    mask_bulk: bool = False
    interpolate: bool = True

    # eval arguments
    num_eval_cases: int = 50
    autoreg_start: int = 1

if __name__ == "__main__":
    
    #===============#
    cfg = CLI(Config, as_positional=False)
    #===============#

    if not (cfg.analysis or cfg.extraction or cfg.visualization or cfg.train or cfg.eval):
        print("No mode selected. Select one of analysis, extraction, visualization, train, eval")
        exit()

    DISTRIBUTED = mlutils.is_torchrun()
    LOCAL_RANK = int(os.environ['LOCAL_RANK']) if DISTRIBUTED else 0
    device = mlutils.select_device()

    #===============#
    mlutils.set_seed(cfg.seed)
    #===============#

    if cfg.analysis:
        if LOCAL_RANK == 0:
            am.compute_dataset_statistics(PROJDIR, DATADIR_FINALTIME, SUBDIRS)
            am.make_exclusion_list(PROJDIR)
            am.compute_filtered_dataset_statistics(PROJDIR)
        exit()

    if cfg.extraction:
        do_extraction()
        exit()

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    if cfg.train or cfg.visualization: # create a new experiment directory
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

    if cfg.eval: # load config from experiment directory, then write to case_dir/final
        assert os.path.exists(case_dir)
        config_file = os.path.join(case_dir, 'config.yaml')
        _cfg = cfg
        if LOCAL_RANK == 0:
            print(f'Loading config from {config_file}')
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)

        cfg = Config(**cfg)
        cfg.eval = True
        cfg.train = False
        cfg.autoreg_start = _cfg.autoreg_start
        cfg.num_eval_cases = _cfg.num_eval_cases

    if DISTRIBUTED:
        torch.distributed.barrier()

    if cfg.visualization:
        if cfg.timeseries:
            vis_timeseries(cfg)
        else:
            vis_finaltime(cfg)
        exit()

    # train or eval
    if cfg.timeseries:
        train_timeseries(cfg, device)
    else:
        train_finaltime(cfg, device)

    #===============#
    mlutils.dist_finalize()
    #===============#

    exit()
#