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
    DATADIR_BASE = '/mnt/hdd1/vedantpu/data/'
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
def main(cfg, device):
    DISTRIBUTED = mlutils.is_torchrun()
    GLOBAL_RANK = int(os.environ['RANK']) if DISTRIBUTED else 0

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    #=================#
    # DATA
    #=================#
    
    _data, data_ = bench.load_dataset(cfg.dataset, DATADIR_BASE)
    
    if cfg.dataset in ['elasticity', 'darcy']:
        c_in = 2
        c_out = 1
        cond = False
    elif cfg.dataset in ['airfoil', 'cylinder_flow']:
        c_in = 2
        c_edge = 2
        c_out = 2
        cond = True
    else:
        print(f"Dataset {cfg.dataset} not found.")
        exit()

    if GLOBAL_RANK == 0:
        print(f"Loaded {cfg.dataset} dataset with {len(_data)} train and {len(data_)} test cases.")
    
    #=================#
    # MODEL
    #=================#

    if cond:
        if cfg.model_type == 0:
            model = bench.Transolver(
                space_dim=c_in+2, out_dim=c_out, fun_dim=0,
                n_hidden=cfg.width, n_layers=cfg.num_layers,
                n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                slice_num=cfg.num_slices,
            )
        elif cfg.model_type == 1:
            model = bench.TS1(
                in_dim=c_in, out_dim=c_out,
                n_hidden=cfg.width, n_layers=cfg.num_layers,
                n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                num_slices=cfg.num_slices,
            )
        elif cfg.model_type == 2:
            model = bench.TS2(
                in_dim=c_in, out_dim=c_out,
                n_hidden=cfg.width, n_layers=cfg.num_layers,
                n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                num_slices=cfg.num_slices,
            )
        elif cfg.model_type == 3:
            model = bench.TS3(
                in_dim=c_in, out_dim=c_out,
                n_hidden=cfg.width, n_layers=cfg.num_layers,
                n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                num_slices=cfg.num_slices,
            )
        elif cfg.model_type == 4:
            model = bench.TS4(
                in_dim=c_in, out_dim=c_out,
                n_hidden=cfg.width, n_layers=cfg.num_layers,
                n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                num_slices=cfg.num_slices,
            )
        else:
            print(f"No conditioned model selected.")
            raise NotImplementedError()
    else:
        if cfg.model_type == 0:
            model = bench.Transolver(
                space_dim=c_in, out_dim=c_out, fun_dim=0,
                n_hidden=cfg.width, n_layers=cfg.num_layers,
                n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                slice_num=cfg.num_slices,
            )
        elif cfg.model_type == 1:
            model = bench.TS1Uncond(
                in_dim=c_in, out_dim=c_out,
                n_hidden=cfg.width, n_layers=cfg.num_layers,
                n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                num_slices=cfg.num_slices,
            )
        elif cfg.model_type == 2:
            model = bench.TS2Uncond(
                in_dim=c_in, out_dim=c_out,
                n_hidden=cfg.width, n_layers=cfg.num_layers,
                n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                num_slices=cfg.num_slices,
            )
        elif cfg.model_type == 999:
            model = am.MeshGraphNet(c_in, c_edge, c_out, cfg.width, cfg.num_layers)
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
        
        # L2 = torch.nn.MSELoss()
        # alpha = 1e-0
        # def batch_lossfun(trainer, model, batch):
        #     x, y = batch
        #     yh, slice_weights, temperature, attn_weights = model(x, return_stats=True)

        #     l2_loss = L2(yh, y)
        #     l1_loss = sum(w.abs().sum() / w.numel() for w in slice_weights) / len(slice_weights)

        #     loss = l2_loss + alpha * l1_loss
        #     return loss
        
        # kw['batch_lossfun'] = batch_lossfun
        
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
    model_type: int = 0 # 0: Transolver, 1: TS1, ..., 999: MeshGraphNet
    width: int = 128
    num_layers: int = 5
    num_heads: int = 8
    mlp_ratio: float = 2.0
    num_slices: int = 32

    # training arguments
    epochs: int = 100
    batch_size: int = 1
    weight_decay: float = 1e-3
    learning_rate: float = 1e-3
    schedule: str = None
    one_cycle_pct_start:float = 0.3
    one_cycle_div_factor: float = 25
    one_cycle_final_div_factor: float = 1e4
    one_cycle_three_phase: bool = True
    

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

    main(cfg, device)

    #===============#
    mlutils.dist_finalize()
    #===============#

    exit()
#