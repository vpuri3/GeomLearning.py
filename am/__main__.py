#
import torch

import os
import yaml
from tqdm import tqdm
from jsonargparse import CLI
from dataclasses import dataclass

# local
import am
import mlutils

#======================================================================#
import socket
MACHINE = socket.gethostname()

if MACHINE == "eagle":
    # VDEL Eagle - 1 node: 4x 2080Ti
    PROJDIR      = '/home/vedantpu/.julia/dev/GeomLearning.py'
    DATADIR_BASE = '/mnt/hdd1/vedantpu/data/NetFabb/'
elif MACHINE.startswith("gpu-node-"):
    # MAIL GPU - 1 node: 8x 2080Ti
    PROJDIR      = '/home/vedantpu/GeomLearning.py'
    DATADIR_BASE = '/home/vedantpu/GeomLearning.py/data/'
elif MACHINE.startswith("v"):
    # PSC Bridges - 8x v100 32GB
    PROJDIR      = "/ocean/projects/eng170006p/vpuri1/GeomLearning.py"
    DATADIR_BASE = 'data/'

#======================================================================#

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

PROJDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CASEDIR = os.path.join(PROJDIR, 'out', 'am')
os.makedirs(CASEDIR, exist_ok=True)

#======================================================================#
def train_timeseries(cfg, device):
    DISTRIBUTED = mlutils.is_torchrun()
    GLOBAL_RANK = int(os.environ['RANK']) if DISTRIBUTED else 0

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    #=================#
    # DATA
    #=================#

    # read in exclusion list
    exclude_list = os.path.join(PROJDIR, 'analysis', 'exclusion_list.txt')
    exclude_list = [line.strip() for line in open(exclude_list, 'r').readlines()]
    
    transform = am.TimeseriesDatasetTransform(
        disp=cfg.disp, vmstr=cfg.vmstr, temp=cfg.temp,
        sdf=cfg.sdf, mesh=cfg.GNN, metadata=False,
        merge=cfg.merge, interpolate=cfg.interpolate,
    )
    DATADIRS = [os.path.join(DATADIR_TIMESERIES, DIR) for DIR in SUBDIRS]
    DATADIRS = DATADIRS[:5]
    dataset = am.TimeseriesDataset(
        DATADIRS, merge=cfg.merge, exclude_list=exclude_list,
        transform=transform, verbose=GLOBAL_RANK==0,
        # force_reload=True,
    )

    # _data, data_ = am.split_timeseries_dataset(dataset, split=[0.8, 0.2])
    _data, data_, _ = am.split_timeseries_dataset(dataset, split=[0.2, 0.05, 0.75])
    # _data, data_, _ = am.split_timeseries_dataset(dataset, split=[0.05, 0.01, 0.94])
    
    if GLOBAL_RANK == 0:
        print(f"Loaded {len(dataset.case_files)} cases from {DATADIR_TIMESERIES}")
        print(f"Split into {len(_data.case_files)} train and {len(data_.case_files)} test cases")
    
    #=================#
    # MODEL
    #=================#

    ci = 3 + (cfg.disp + cfg.vmstr + cfg.temp) + (cfg.sdf * 10) # (pos, fields, sdf)
    ce = 3
    co = cfg.disp + cfg.vmstr + cfg.temp

    if cfg.GNN:
        model = am.MeshGraphNet(ci, ce, co, cfg.gnn_width, cfg.gnn_num_layers)
    elif cfg.TRA == 0:
        model = am.Transolver(
            space_dim=ci+2, out_dim=co, fun_dim=0,
            n_hidden=cfg.tra_width, n_layers=cfg.tra_num_layers,
            n_head=cfg.tra_num_heads, mlp_ratio=cfg.tra_mlp_ratio,
            slice_num=cfg.tra_num_slices,
        )
    elif cfg.TRA == 1:
        model = am.TS1(
            in_dim=ci, out_dim=co,
            n_hidden=cfg.tra_width, n_layers=cfg.tra_num_layers,
            n_head=cfg.tra_num_heads, mlp_ratio=cfg.tra_mlp_ratio,
            num_slices=cfg.tra_num_slices,
        )
    elif cfg.TRA == 2:
        model = am.TS2(
            in_dim=ci, out_dim=co,
            n_hidden=cfg.tra_width, n_layers=cfg.tra_num_layers,
            n_head=cfg.tra_num_heads, mlp_ratio=cfg.tra_mlp_ratio,
            num_slices=cfg.tra_num_slices,
        )
    elif cfg.TRA == 3:
        model = am.TS3(
            in_dim=ci, out_dim=co,
            n_hidden=cfg.tra_width, n_layers=cfg.tra_num_layers,
            n_head=cfg.tra_num_heads, mlp_ratio=cfg.tra_mlp_ratio,
            num_slices=cfg.tra_num_slices,
        )
    else:
        print(f"No model selected. Choose between GNN or TRA.")
        raise NotImplementedError()

    model = am.MaskedModel(model, mask=cfg.mask, mask_bulk=cfg.mask_bulk)
    
    #=================#
    # TRAIN
    #=================#

    batch_lossfun = am.MaskedLoss(cfg.mask)
    callback = am.TimeseriesCallback(case_dir, mesh=cfg.GNN, num_eval_cases=cfg.num_eval_cases, autoreg_start=cfg.autoreg_start)

    if cfg.train and cfg.epochs > 0:

        _batch_size = batch_size_ = _batch_size_ = 1

        kw = dict(
            device=device, gnn_loader=True, stats_every=cfg.epochs//10,
            Opt='AdamW', weight_decay=cfg.weight_decay, epochs=cfg.epochs,
            _batch_size=_batch_size, batch_size_=batch_size_, _batch_size_=_batch_size_,
            batch_lossfun=batch_lossfun,
        )
        
        # scheduler
        if cfg.schedule is None or cfg.schedule == 'ConstantLR':
            kw = dict(
                **kw,
                lr=cfg.learning_rate,
            )
        elif cfg.schedule == 'OneCycleLR':
            kw = dict(
                **kw,
                Schedule='OneCycleLR',
                lr = cfg.learning_rate,
                one_cycle_pct_start=cfg.one_cycle_pct_start,
                one_cycle_div_factor=cfg.one_cycle_div_factor,
                one_cycle_final_div_factor=cfg.one_cycle_final_div_factor,
                one_cycle_three_phase=cfg.one_cycle_three_phase,
            )

        trainer = mlutils.Trainer(model, _data, data_, **kw)
        trainer.add_callback('epoch_end', callback)

        if cfg.restart_file is not None:
            trainer.load(cfg.restart_file)

        trainer.train()

    #=================#
    # ANALYSIS
    #=================#

    # if device != 'cpu' and device != torch.device('cpu'):
    #     torch.cuda.empty_cache()
    # trainer = mlutils.Trainer(model, _data, data_, device=device)
    # callback.load(trainer)
    # callback(trainer, final=True)

    return

#======================================================================#
def train_finaltime(cfg, device):
    DISTRIBUTED = mlutils.is_torchrun()
    GLOBAL_RANK = int(os.environ['RANK']) if DISTRIBUTED else 0

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
        
    transform = am.FinaltimeDatasetTransform(
        disp=cfg.disp, vmstr=cfg.vmstr, temp=cfg.temp,
        sdf=cfg.sdf, mesh=cfg.GNN,
    )
    dataset = am.CompositeDataset(*datasets, transform=transform)
    # _data, data_ = torch.utils.data.random_split(dataset, [0.80, 0.20])
    split = int(len(dataset) * 0.80)
    indices = [split, len(dataset) - split]
    _data, data_ = torch.utils.data.random_split(dataset, indices)
    
    # split = int(len(dataset) * 0.10)
    # indices = [split, 8, len(dataset) - split - 8]
    # _data, data_, _ = torch.utils.data.random_split(dataset, indices)
    
    if GLOBAL_RANK == 0:
        print(f"Loaded {len(dataset)} cases from {DATADIR_FINALTIME}")
        print(f"Split into {len(_data)} train and {len(data_)} test cases")
    
    #=================#
    # MODEL
    #=================#

    ci = 3 + (cfg.sdf * 10)
    ce = 3
    co = cfg.disp + cfg.vmstr + cfg.temp

    if cfg.GNN:
        model = am.MeshGraphNet(ci, ce, co, cfg.gnn_width, cfg.gnn_num_layers)
    elif cfg.TRA == 0:
        model = am.Transolver(
            space_dim=ci, out_dim=co, fun_dim=0,
            n_hidden=cfg.tra_width, n_layers=cfg.tra_num_layers,
            n_head=cfg.tra_num_heads, mlp_ratio=cfg.tra_mlp_ratio,
            slice_num=cfg.tra_num_slices,
        )
    elif cfg.TRA == 1:
        model = am.TS1Uncond(
            in_dim=ci, out_dim=co,
            n_hidden=cfg.tra_width, n_layers=cfg.tra_num_layers,
            n_head=cfg.tra_num_heads, mlp_ratio=cfg.tra_mlp_ratio,
            num_slices=cfg.tra_num_slices,
        )
    elif cfg.TRA == 2:
        model = am.TS2Uncond(
            in_dim=ci, out_dim=co,
            n_hidden=cfg.tra_width, n_layers=cfg.tra_num_layers,
            n_head=cfg.tra_num_heads, mlp_ratio=cfg.tra_mlp_ratio,
            num_slices=cfg.tra_num_slices,
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

        _batch_size = batch_size_ = _batch_size_ = 1

        kw = dict(
            device=device, gnn_loader=True, stats_every=cfg.epochs//10,
            Opt='AdamW', weight_decay=cfg.weight_decay, lossfun=lossfun, epochs=cfg.epochs,
            _batch_size=_batch_size, batch_size_=batch_size_, _batch_size_=_batch_size_
        )

        # scheduler
        if cfg.schedule is None or cfg.schedule == 'ConstantLR':
            kw = dict(
                **kw,
                lr=cfg.learning_rate,
            )
        elif cfg.schedule == 'OneCycleLR':
            kw = dict(
                **kw,
                Schedule='OneCycleLR',
                lr = cfg.learning_rate,
                one_cycle_pct_start=cfg.one_cycle_pct_start,
                one_cycle_div_factor=cfg.one_cycle_div_factor,
                one_cycle_final_div_factor=cfg.one_cycle_final_div_factor,
                one_cycle_three_phase=cfg.one_cycle_three_phase,
            )

        trainer = mlutils.Trainer(model, _data, data_, **kw)
        trainer.add_callback('epoch_end', callback)

        if cfg.restart_file is not None:
            trainer.load(cfg.restart_file)

        trainer.train()

    #=================#
    # ANALYSIS
    #=================#

    # if device != 'cpu' and device != torch.device('cpu'):
    #     torch.cuda.empty_cache()
    # trainer = mlutils.Trainer(model, _data, data_, device=device)
    # callback.load(trainer)
    # callback(trainer, final=True)

    return

#======================================================================#
def vis_finaltime(cfg, force_reload=True, max_cases=50, num_workers=None):

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    # read in exclusion list
    exclude_list = os.path.join(PROJDIR, 'analysis', 'exclusion_list.txt')
    exclude_list = [line.strip() for line in open(exclude_list, 'r').readlines()]
    
    # for DIR in SUBDIRS[:1]:
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

        for icase in tqdm(range(num_cases), ncols=80):
            data = dataset[icase]
            ii = str(icase).zfill(3)
            case_name = data.metadata['case_name']
            out_file = os.path.join(vis_dir, f'{ii}_{case_name}.vtu')
            am.visualize_pyv(data, out_file)

    return

#======================================================================#
def vis_timeseries(cfg, force_reload=False, max_cases=50, num_workers=None):

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    # read in exclusion list
    exclude_list = os.path.join(PROJDIR, 'analysis', 'exclusion_list.txt')
    exclude_list = [line.strip() for line in open(exclude_list, 'r').readlines()]

    # for DIR in SUBDIRS[:1]:
    for DIR in SUBDIRS[:10]:
        DATADIR = os.path.join(DATADIR_TIMESERIES, DIR)
        dataset = am.TimeseriesDataset(
            DATADIR,
            merge=cfg.merge,
            exclude_list=exclude_list,
            num_workers=num_workers,
            force_reload=force_reload,
        )
        vis_dir = os.path.join(case_dir, DIR)
        os.makedirs(vis_dir, exist_ok=False)
    
        case_names = [os.path.basename(f)[:-3] for f in dataset.case_files]
        num_cases  = min(len(case_names), max_cases)

        for icase in tqdm(range(num_cases), ncols=80):
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
    for DIR in SUBDIRS[5:10]:
        zip_file = os.path.join(DATADIR_RAW, DIR + ".zip")

        out_dir  = os.path.join(DATADIR_TIMESERIES, DIR)
        am.extract_from_zip(zip_file, out_dir, timeseries=True)

        # out_dir  = os.path.join(DATADIR_FINALTIME, DIR)
        # am.extract_from_zip(zip_file, out_dir, timeseries=False)

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
    restart_file: str = None

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
    TRA: int = 0 # 0: Transolver, 1: TS1

    # GNN
    gnn_width: int = 128
    gnn_num_layers: int = 5

    # TRA
    tra_width: int = 128
    tra_num_layers: int = 5
    tra_num_heads: int = 8
    tra_mlp_ratio: float = 2.0
    tra_num_slices: int = 32

    # training arguments
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    schedule: str = None
    one_cycle_pct_start:float = 0.3
    one_cycle_div_factor: float = 25
    one_cycle_final_div_factor: float = 1e4
    one_cycle_three_phase: bool = True

    # timeseries  dataset
    merge: bool = True
    mask: bool = True
    blend: bool = False
    mask_bulk: bool = False
    interpolate: bool = False

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
    GLOBAL_RANK = int(os.environ['RANK']) if DISTRIBUTED else 0
    device = mlutils.select_device()

    #===============#
    mlutils.set_seed(cfg.seed)
    #===============#

    if cfg.analysis:
        if GLOBAL_RANK == 0:
            am.compute_dataset_statistics(PROJDIR, DATADIR_FINALTIME, SUBDIRS)
            am.make_exclusion_list(PROJDIR)
            am.compute_filtered_dataset_statistics(PROJDIR)
        exit()

    if cfg.extraction:
        do_extraction()
        exit()

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    # create a new experiment directory
    if cfg.train or cfg.visualization:
        if os.path.exists(case_dir):
            nd = len([dir for dir in os.listdir(CASEDIR) if dir.startswith(cfg.exp_name)])
            cfg.exp_name = cfg.exp_name + '_' + str(nd).zfill(2)
            case_dir = os.path.join(CASEDIR, cfg.exp_name)

        if DISTRIBUTED:
            torch.distributed.barrier()

        if GLOBAL_RANK == 0:
            os.makedirs(case_dir)
            config_file = os.path.join(case_dir, 'config.yaml')
            print(f'Saving config to {config_file}')
            with open(config_file, 'w') as f:
                yaml.safe_dump(vars(cfg), f)

    # load config from experiment directory
    if cfg.eval:
        assert os.path.exists(case_dir)
        config_file = os.path.join(case_dir, 'config.yaml')
        _cfg = cfg
        if GLOBAL_RANK == 0:
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
