#
import torch

import os
import yaml
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
    
    _data, data_ = bench.load_dataset(
        cfg.dataset,
        DATADIR_BASE,
        force_reload=cfg.force_reload,
        mesh=cfg.model_type == -1,
        max_cases=cfg.max_cases,
        max_steps=cfg.max_steps,
    )
    
    if cfg.dataset in ['elasticity', 'darcy']:
        c_in = 2
        c_out = 1
        time_cond = False

        if GLOBAL_RANK == 0:
            print(f"Loaded {cfg.dataset} dataset with {len(_data)} train and {len(data_)} test cases.")
    
    elif cfg.dataset in ['airfoil', 'cylinder_flow']:
        
        c_in = 11  # node_type (7) + pos (2) + vel0 (2)
        c_edge = 2 # x, y
        c_out = 2  # vel1 (2)
        if cfg.dataset == 'airfoil':
            c_in += 1  # density0
            c_out += 1 # density1
        
        time_cond = True

        if GLOBAL_RANK == 0:
            print(f"Loaded {cfg.dataset} dataset with {_data.num_cases} train and {data_.num_cases} test cases.")
            print(f"Number of time-steps: {_data.trajectory_length}")
            
            if cfg.max_steps is not None:
                print(f"Limiting to {cfg.max_steps} time-steps")
            if cfg.max_cases is not None:
                print(f"Limiting to {cfg.max_cases} cases")
    
            for graph in _data:
                print(graph)
                break

    else:
        print(f"Dataset {cfg.dataset} not found.")
        exit()
        
    #=================#
    # MODEL
    #=================#

    if time_cond:
        if cfg.model_type == -1:
            model = am.MeshGraphNet(c_in, c_edge, c_out, cfg.hidden_dim, cfg.num_layers)
        elif cfg.model_type == 0:
            model = am.Transolver(
                space_dim=c_in+1, out_dim=c_out, fun_dim=0,
                n_hidden=cfg.hidden_dim, n_layers=cfg.num_layers,
                n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                slice_num=cfg.num_slices,
            )
        elif cfg.model_type == 1:
            model = am.TS1( # Physics attention + AdaLN conditioning
                in_dim=c_in, out_dim=c_out,
                n_hidden=cfg.hidden_dim, n_layers=cfg.num_layers,
                n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                num_slices=cfg.num_slices,
            )
        elif cfg.model_type == 2:
            model = am.TS2( # Slice attention + AdaLN conditioning (query size [M, D])
                in_dim=c_in, out_dim=c_out,
                n_hidden=cfg.hidden_dim, n_layers=cfg.num_layers,
                n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                num_slices=cfg.num_slices,
            )
        elif cfg.model_type == 3:
            model = am.TS3( # Slice attention + slice query conditioning (query size [M, D])
                in_dim=c_in, out_dim=c_out,
                n_hidden=cfg.hidden_dim, n_layers=cfg.num_layers,
                n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                num_slices=cfg.num_slices,
            )
        else:
            raise NotImplementedError("No time-conditioned model selected.")
    else:
        if cfg.model_type == -1:
            model = am.MeshGraphNet(c_in, c_edge, c_out, cfg.hidden_dim, cfg.num_layers)
        elif cfg.model_type == 0:
            model = bench.Transolver(
                space_dim=c_in, out_dim=c_out, fun_dim=0,
                n_hidden=cfg.hidden_dim, n_layers=cfg.num_layers,
                n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                slice_num=cfg.num_slices,
            )
        elif cfg.model_type == 1:
            model = bench.TS1Uncond(
                in_dim=c_in, out_dim=c_out,
                hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
                num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
                num_slices=cfg.num_slices,
                k_val=cfg.topk,
            )
        else:
            raise NotImplementedError("No unconditioned model selected.")
        
    # Use masked model for timeseries datasets
    if cfg.dataset in ['airfoil', 'cylinder_flow']:
        if GLOBAL_RANK == 0:
            print(f"Using masked model for timeseries datasets {cfg.dataset}")
        model = am.MaskedModel(model, mask=True, reduce_graph=False)

    #=================#
    # TRAIN
    #=================#

    callback = bench.Callback(case_dir,)
    if cfg.model_type in [1,] and (time_cond == False):
        callback = bench.TSCallback(case_dir,)
    if cfg.dataset in ['airfoil', 'cylinder_flow']:
        callback = bench.TimeseriesCallback(case_dir, mesh=cfg.model_type == -1)
        
    if cfg.train and cfg.epochs > 0:

        _batch_size  = cfg.batch_size
        if cfg.dataset == 'airfoil':
            batch_size_ = _batch_size_ = 50
        elif cfg.dataset == 'cylinder_flow':
            batch_size_ = _batch_size_ = 50
        elif cfg.dataset == 'elasticity':
            batch_size_ = _batch_size_ = 200
        
        lossfun = torch.nn.MSELoss()
        gnn_loader = cfg.dataset in ['airfoil', 'cylinder_flow']

        kw = dict(
            device=device, gnn_loader=gnn_loader, stats_every=cfg.epochs//10,
            Opt='AdamW', weight_decay=cfg.weight_decay, epochs=cfg.epochs,
            _batch_size=_batch_size, batch_size_=batch_size_, _batch_size_=_batch_size_,
            lossfun=lossfun, clip_grad_norm=1.,
        )
        
        # LR scheduler
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
            
        # Noise schedule
        kw['noise_schedule'] = cfg.noise_schedule
        kw['noise_init'] = cfg.noise_init

        #-------------#
        # make Trainer
        #-------------#

        trainer = mlutils.Trainer(model, _data, data_, **kw)
        trainer.add_callback('epoch_end', callback)

        #-------------#
        # batch_lossfun
        #-------------#
        if (cfg.model_type == 1) and cfg.dataset == 'elasticity':
            gamma_schedule = mlutils.DecayScheduler(
                init_val=cfg.gamma_init, min_val=cfg.gamma_min,
                total_steps=trainer.total_steps // 2,
                decay_type=cfg.gamma_schedule,
            )

            def batch_lossfun(trainer, model, batch):
                x, y = batch
                if model.training:
                    gamma_schedule.step()
                    gamma = gamma_schedule.get_current_val()
                    yh = model(x, gamma=gamma)
                else:
                    yh = model(x)
                return lossfun(yh, y)

            trainer.batch_lossfun = batch_lossfun
            
        if cfg.dataset in ['airfoil', 'cylinder_flow']:
            if GLOBAL_RANK == 0:
                print(f"Using masked loss for timeseries datasets {cfg.dataset}")
            batch_lossfun = am.MaskedLoss(mask=True)
            trainer.batch_lossfun = batch_lossfun

        #-------------#
        # load snapshot and train
        #-------------#

        if cfg.restart_file is not None:
            trainer.load(cfg.restart_file)

        trainer.train()

    #=================#
    # ANALYSIS
    #=================#
    
    if cfg.eval:
        if device != 'cpu' and device != torch.device('cpu'):
            torch.cuda.empty_cache()
        trainer = mlutils.Trainer(model, _data, data_, device=device)
        callback.load(trainer)
        callback(trainer, final=True)

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
    restart_file: str = None
    exp_name: str = 'exp'
    seed: int = 0

    # dataset
    dataset: str = None
    force_reload: bool = False
    max_cases: int = 10 # only used in cylinder_flow and airfoil
    max_steps: int = 50

    # model
    model_type: int = 0 # 0: Transolver, 1: TS1, ..., -1: MeshGraphNet
    act: str = 'gelu'
    hidden_dim: int = 128
    num_layers: int = 5
    num_heads: int = 8
    mlp_ratio: float = 2.0
    num_slices: int = 32

    # training arguments
    epochs: int = 100
    batch_size: int = 1
    weight_decay: float = 0e-0
    learning_rate: float = 1e-3
    schedule: str = None
    one_cycle_pct_start:float = 0.1
    one_cycle_div_factor: float = 25
    one_cycle_final_div_factor: float = 40
    one_cycle_three_phase: bool = False

    topk: int = 0
    gamma_min: float = 0e-4
    gamma_init: float = 1e-2
    gamma_schedule: str = 'constant'
    noise_init: float = 1e-1
    noise_schedule: str = 'linear'

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