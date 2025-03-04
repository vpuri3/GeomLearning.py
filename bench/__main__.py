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
import bench

#======================================================================#
import socket
MACHINE = socket.gethostname()

if MACHINE == "eagle":
    # VDEL Eagle - 1 node: 4x 2080Ti
    DATADIR_BASE = '/mnt/hdd1/vedantpu/data/GeoFNO/'
elif MACHINE.startswith("gpu-node-"):
    # MAIL GPU - 1 node: 8x 2080Ti
    DATADIR_BASE = '/home/vedantpu/GeomLearning.py/data/'
elif MACHINE.startswith("v"):
    # PSC Bridges - 8x v100 32GB
    DATADIR_BASE = 'data/'

#======================================================================#
PROJDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CASEDIR = os.path.join(PROJDIR, 'out', 'bench')
os.makedirs(CASEDIR, exist_ok=True)

#======================================================================#
def train(cfg, device):
    DISTRIBUTED = mlutils.is_torchrun()
    GLOBAL_RANK = int(os.environ['RANK']) if DISTRIBUTED else 0

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    #=================#
    # DATA
    #=================#
    
    if cfg.dataset == 'elasticity':
        import numpy as np

        DATADIR = os.path.join(DATADIR_BASE, 'elasticity')
        PATH_Sigma = os.path.join(DATADIR, 'Meshes', 'Random_UnitCell_sigma_10.npy')
        PATH_XY = os.path.join(DATADIR, 'Meshes', 'Random_UnitCell_XY_10.npy')

        input_s = np.load(PATH_Sigma)
        input_s = torch.tensor(input_s, dtype=torch.float).permute(1, 0).unsqueeze(-1)
        input_xy = np.load(PATH_XY)
        input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2, 0, 1)
        
        print(input_xy.shape, input_s.shape)
        
        ntrain = 1000
        ntest  = 200
        
        y_normalizer = bench.UnitTransformer(input_s[:ntrain])
        input_s = y_normalizer.encode(input_s)

        dataset = torch.utils.data.TensorDataset(input_xy, input_s)
        _data = torch.utils.data.Subset(dataset, range(ntrain))
        data_ = torch.utils.data.Subset(dataset, range(len(dataset)-ntest, len(dataset)))

        ci = 2
        co = 1

    else:
        print(f"Dataset {cfg.dataset} not found.")
        exit()

    # _data, data_ = torch.utils.data.random_split(dataset, [0.80, 0.20])
    
    if GLOBAL_RANK == 0:
        print(f"Loaded {cfg.dataset} dataset with {len(dataset)} cases.")
        print(f"Split into {len(_data)} train and {len(data_)} test cases.")
    
    #=================#
    # MODEL
    #=================#

    if cfg.cond:
        if cfg.TRA == 0:
            model = bench.Transolver(
                space_dim=ci+2, out_dim=co, fun_dim=0,
                n_hidden=cfg.width, n_layers=cfg.num_layers,
                n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                slice_num=cfg.num_slices,
            )
        elif cfg.TRA == 1:
            model = bench.TS1(
                in_dim=ci, out_dim=co,
                n_hidden=cfg.width, n_layers=cfg.num_layers,
                n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                num_slices=cfg.num_slices,
            )
        elif cfg.TRA == 2:
            model = bench.TS2(
                in_dim=ci, out_dim=co,
                n_hidden=cfg.width, n_layers=cfg.num_layers,
                n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                num_slices=cfg.num_slices,
            )
        elif cfg.TRA == 3:
            model = bench.TS3(
                in_dim=ci, out_dim=co,
                n_hidden=cfg.width, n_layers=cfg.num_layers,
                n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                num_slices=cfg.num_slices,
            )
        elif cfg.TRA == 4:
            model = bench.TS4(
                in_dim=ci, out_dim=co,
                n_hidden=cfg.width, n_layers=cfg.num_layers,
                n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                num_slices=cfg.num_slices,
            )
        else:
            print(f"No conditioned model selected.")
            raise NotImplementedError()
    else:
        if cfg.TRA == 0:
            model = bench.Transolver(
                space_dim=ci, out_dim=co, fun_dim=0,
                n_hidden=cfg.width, n_layers=cfg.num_layers,
                n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                slice_num=cfg.num_slices,
            )
        elif cfg.TRA == 1:
            model = bench.TS1Uncond(
                in_dim=ci, out_dim=co,
                n_hidden=cfg.width, n_layers=cfg.num_layers,
                n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                num_slices=cfg.num_slices,
            )
        elif cfg.TRA == 2:
            model = bench.TS2Uncond(
                in_dim=ci, out_dim=co,
                n_hidden=cfg.width, n_layers=cfg.num_layers,
                n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                num_slices=cfg.num_slices,
            )
        else:
            print(f"No unconditioned model selected.")
            raise NotImplementedError()

    #=================#
    # TRAIN
    #=================#

    lossfun  = torch.nn.MSELoss()
    callback = bench.Callback(case_dir,)

    if cfg.train and cfg.epochs > 0:

        _batch_size  = cfg.batch_size
        batch_size_  = len(data_)
        _batch_size_ = len(_data)

        kw = dict(
            device=device, gnn_loader=False, stats_every=cfg.epochs//10,
            Opt='AdamW', weight_decay=cfg.weight_decay, epochs=cfg.epochs,
            _batch_size=_batch_size, batch_size_=batch_size_, _batch_size_=_batch_size_,
            lossfun=lossfun,
            clip_grad=1.,
        )
        
        L2 = torch.nn.MSELoss()
        alpha = 1e-0
        def batch_lossfun(trainer, model, batch):
            x, y = batch
            yh, slice_weights, temperature, attn_weights = model(x, return_stats=True)

            l2_loss = L2(yh, y)
            l1_loss = sum(w.abs().sum() / w.numel() for w in slice_weights) / len(slice_weights)

            loss = l2_loss + alpha * l1_loss
            return loss
        
        kw['batch_lossfun'] = batch_lossfun
        
        # scheduler
        if cfg.schedule is None or cfg.schedule == 'ConstantLR':
            kw['lr'] = cfg.learning_rate
        elif cfg.schedule == 'OneCycleLR':
            kw['Schedule'] = 'OneCycleLR'
            kw['lr'] = cfg.learning_rate
            kw['one_cycle_pct_start'] = cfg.one_cycle_pct_start
            kw['one_cycle_div_factor'] = cfg.one_cycle_div_factor
            kw['one_cycle_final_div_factor'] = cfg.one_cycle_final_div_factor
            kw['one_cycle_three_phase'] = cfg.one_cycle_three_phase
        else:
            kw = dict(**kw, Schedule=cfg.schedule, lr=cfg.learning_rate,)

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
@dataclass
class Config:
    '''
    Benchmarks transformer tokenizer models on standard datasets
    '''

    # case configuration
    train: bool = False
    eval: bool = False
    cond: bool = False
    restart_file: str = None
    dataset: str = None

    exp_name: str = 'exp'
    seed: int = 123

    # model
    TRA: int = 0 # 0: Transolver, 1: TS1, ...
    width: int = 128
    num_layers: int = 5
    num_heads: int = 8
    mlp_ratio: float = 2.0
    num_slices: int = 32

    # training arguments
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    schedule: str = None
    one_cycle_pct_start:float = 0.3
    one_cycle_div_factor: float = 25
    one_cycle_final_div_factor: float = 1e4
    one_cycle_three_phase: bool = True
    
    batch_size: int = 1

if __name__ == "__main__":
    
    DISTRIBUTED = mlutils.is_torchrun()
    GLOBAL_RANK = int(os.environ['RANK']) if DISTRIBUTED else 0
    device = mlutils.select_device()

    #===============#
    cfg = CLI(Config, as_positional=False)
    #===============#

    if not (cfg.train or cfg.eval):
        print("No mode selected. Select one of train, eval")
        exit()

    if cfg.dataset is None:
        print("No dataset selected.")
        exit()

    #===============#
    mlutils.set_seed(cfg.seed)
    #===============#

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    # create a new experiment directory
    if cfg.train:
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

    if DISTRIBUTED:
        torch.distributed.barrier()

    train(cfg, device)

    #===============#
    mlutils.dist_finalize()
    #===============#

    exit()
#