#
import torch

import os
import yaml
import shutil
from tqdm import tqdm
from jsonargparse import ArgumentParser, CLI
from dataclasses import dataclass

# local
import am
import mlutils

# DATADIR_BASE       = 'data/'
DATADIR_BASE       = '/home/shared/'
DATADIR_RAW        = os.path.join(DATADIR_BASE, 'netfabb_ti64_hires_raw')
DATADIR_TIMESERIES = os.path.join(DATADIR_BASE, 'netfabb_ti64_hires_timeseries')
DATADIR_FINALTIME  = os.path.join(DATADIR_BASE, 'netfabb_ti64_hires_finaltime')

CASEDIR = os.path.join('.', 'out', 'am')

#======================================================================#
def train_timeseries(cfg, device):
    DISTRIBUTED = mlutils.is_torchrun()
    LOCAL_RANK = int(os.environ['LOCAL_RANK']) if DISTRIBUTED else 0

    case_dir = os.path.join(CASEDIR, cfg.name)

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

    # run small experiments
    N1 = 30
    N2 = 10
    _data, data_ = am.split_timeseries_dataset(dataset, indices=[range(N1), range(N1,N1+N2)])

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

    callback = am.TimeseriesCallback(case_dir, cfg.GNN, autoreg_start=cfg.autoreg_start, num_eval_cases=cfg.num_eval_cases)
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
                Opt='AdamW', weight_decay=cfg.weight_decay, lossfun=lossfun,
                _batch_size=_batch_size, batch_size_=batch_size_, _batch_size_=_batch_size_,
                batch_lossfun=batch_lossfun,
            )

            kw = dict(lr=5e-4, epochs=cfg.epochs, **kw,)
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

    case_dir = os.path.join(CASEDIR, cfg.name)

    #=================#
    # DATA
    #=================#

    DATADIR = os.path.join(DATADIR_FINALTIME, r"data_0-100")
    DATADIR = os.path.join(DATADIR_FINALTIME, r"data_100-200")

    transform = am.FinaltimeDatasetTransform(disp=cfg.disp, vmstr=cfg.vmstr, temp=cfg.temp, mesh=cfg.GNN)
    dataset = am.FinaltimeDataset(DATADIR, transform=transform)#, force_reload=True)
    _data, data_ = torch.utils.data.random_split(dataset, [0.8, 0.2])

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
            # slice_num=32,
        )
    else:
        raise NotImplementedError()

    #=================#
    # TRAIN
    #=================#

    lossfun  = torch.nn.MSELoss()
    callback = am.FinaltimeCallback(case_dir, cfg.GNN)

    if cfg.train:
        if cfg.epochs > 0:
            _batch_size  = 1
            batch_size_  = 1
            _batch_size_ = 1

            kw = dict(
                device=device, gnn_loader=True, stats_every=cfg.epochs//10,
                Opt='AdamW', weight_decay=cfg.weight_decay, lossfun=lossfun,
                _batch_size=_batch_size, batch_size_=batch_size_, _batch_size_=_batch_size_
            )

            kw = dict(lr=5e-4, epochs=cfg.epochs, **kw,)
            # kw = dict(lr=1e-3, Schedule="OneCycleLR", epochs=cfg.epochs, **kw,)
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
def vis_timeseries(cfg, num_workers=12):

    DIRS = [
        # r'data_0-100',
        r'data_100-200',
        r'data_200-300',
        r'data_300-400',
        r'data_400-500',
        r'data_500-600',
    ]

    # for DIR in DIRS:
    #     DATADIR  = os.path.join(DATADIR_TIMESERIES, DIR)
    #     print(DIR)
    #     dataset  = am.TimeseriesDataset(DATADIR, merge=cfg.merge, force_reload=True, num_workers=8)

    for DIR in DIRS:
        DATADIR  = os.path.join(DATADIR_TIMESERIES, DIR)
        dataset  = am.TimeseriesDataset(DATADIR, merge=cfg.merge)
        case_dir = os.path.join(CASEDIR, cfg.name)
        vis_dir  = os.path.join(case_dir, DIR)
    
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
def test_timeseries_extraction():
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

#======================================================================#
@dataclass
class Config:
    '''
    Train grah neural networks on time series AM data
    '''

    # case configuration
    name: str = 'test'
    seed: int = 123
    train: bool = False

    # timeseries dataset
    merge: bool = True

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
    tra_width: int = 192
    tra_num_layers: int = 5
    tra_num_heads: int = 16
    tra_mlp_ratio: float = 2.0

    # dataset
    mask: bool = True
    blend: bool = True
    mask_bulk: bool = False
    interpolate: bool = True

    # training arguments
    epochs: int = 100
    weight_decay: float = 0e-4

    # eval arguments
    num_eval_cases: int = 20
    autoreg_start: int = 10

if __name__ == "__main__":

    #===============#
    cfg = CLI(Config, as_positional=False)
    #===============#

    DISTRIBUTED = mlutils.is_torchrun()
    LOCAL_RANK = int(os.environ['LOCAL_RANK']) if DISTRIBUTED else 0
    device = mlutils.select_device()

    #===============#
    mlutils.set_seed(cfg.seed)
    #===============#

    case_dir = os.path.join(CASEDIR, cfg.name)

    if cfg.train:
        if os.path.exists(case_dir):
            nd = len([dir for dir in os.listdir(CASEDIR) if dir.startswith(cfg.name)])
            case_dir = case_dir + str(nd).zfill(2)
            cfg.name = cfg.name + str(nd).zfill(2)

        config_file = os.path.join(case_dir, 'config.yaml')

        if LOCAL_RANK == 0:
            os.makedirs(case_dir)
            print(f'Saving config to {config_file}')
            with open(config_file, 'w') as f:
                yaml.safe_dump(vars(cfg), f)
    else:
        assert os.path.exists(case_dir)

    if DISTRIBUTED:
        torch.distributed.barrier()

    #===============#
    # Final time data
    #===============#
    # am.extract_zips(DATADIR_RAW, DATADIR_FINALTIME)
    # train_finaltime(cfg, device)

    #===============#
    # Timeseries data
    #===============#
    # test_timeseries_extraction()
    # am.extract_zips(DATADIR_RAW, DATADIR_TIMESERIES, timeseries=True, num_workers=12)
    # vis_timeseries(cfg)
    train_timeseries(cfg, device)

    #===============#
    mlutils.dist_finalize()
    #===============#

    exit()
#
