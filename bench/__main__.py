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

    _data, data_, metadata = bench.load_dataset(
        cfg.dataset,
        DATADIR_BASE,
        PROJDIR,
        force_reload=cfg.force_reload,
        mesh=cfg.model_type in [-1,],
        max_cases=cfg.max_cases,
        max_steps=cfg.max_steps,
        init_step=cfg.init_step,
        init_case=cfg.init_case,
        exclude=cfg.exclude,
        train_rollout_noise=cfg.train_rollout_noise,
    )

    if cfg.dataset in ['elasticity', 'darcy', 'airfoil_steady', 'pipe']:
        c_in = 2 if not cfg.dataset in ['darcy'] else 3
        c_out = 1
        time_cond = False

        if GLOBAL_RANK == 0:
            print(f"Loaded {cfg.dataset} dataset with {len(_data)} train and {len(data_)} test cases.")
            
    elif cfg.dataset in ['plasticity']:
        c_in = 0
        c_out = 0

        time_cond = True
            
    elif cfg.dataset in ['navier_stokes']:
        c_in  = 12
        c_out = 1

        time_cond = True

    elif cfg.dataset in ['shapenet_car']:
        c_in = 3
        c_out = 1

        time_cond = False

    elif cfg.dataset in ['airfoil', 'cylinder_flow']:
        
        c_in = 11  # node_type (7) + pos (2) + vel0 (2)
        c_edge = 2 # x, y
        c_out = 2  # vel1 (2)
        
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
            model = am.MeshGraphNet(c_in, c_edge, c_out, cfg.channel_dim, cfg.num_blocks)
        elif cfg.model_type == 0:
            model = am.Transolver(
                space_dim=c_in+1, out_dim=c_out, fun_dim=0,
                n_hidden=cfg.channel_dim, n_layers=cfg.num_blocks,
                n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio, slice_num=cfg.num_clusters,
            )
        elif cfg.model_type == 1:
            raise NotImplementedError("CAT not implemented for time-conditioned problems.")
        else:
            raise NotImplementedError("No time-conditioned model selected.")
    else:
        if cfg.model_type == -2:
            if GLOBAL_RANK == 0:
                print(f"Using LNO.")
            n_dim = 192 if cfg.dataset == 'elasticity' else 128
            n_layer = 3 if cfg.dataset == 'elasticity' else 2
            model = bench.LNO(
                n_block=4, n_mode=256, n_dim=n_dim, n_head=8, n_layer=n_layer, act="GELU",
                x_dim=c_in, y1_dim=c_in, y2_dim=c_out,
                model_attr={"time": time_cond,}
            )
        elif cfg.model_type == -1:
            model = am.MeshGraphNet(c_in, c_edge, c_out, cfg.channel_dim, cfg.num_blocks)
        elif cfg.model_type == 0:
            if GLOBAL_RANK == 0:
                print(f"Using Transolver with {cfg.channel_dim} channel dim, {cfg.num_blocks} blocks, {cfg.num_heads} heads, {cfg.mlp_ratio} mlp ratio, {cfg.num_clusters} slices")
            if cfg.conv2d:
                model = bench.Transolver_Structured_Mesh_2D(
                    space_dim=c_in, out_dim=c_out, fun_dim=0,
                    n_hidden=cfg.channel_dim, n_layers=cfg.num_blocks,
                    n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio, slice_num=cfg.num_clusters,
                    H=metadata['H'], W=metadata['W'],
                    unified_pos=cfg.unified_pos,
                )
            else:
                model = bench.Transolver(
                    space_dim=c_in, out_dim=c_out, fun_dim=0,
                    n_hidden=cfg.channel_dim, n_layers=cfg.num_blocks,
                    n_head=cfg.num_heads, mlp_ratio=cfg.mlp_ratio, slice_num=cfg.num_clusters,
                )
        elif cfg.model_type == 1:
            if GLOBAL_RANK == 0:
                print(
                    f"Using CAT with\n" +
                    f"channel_dim={cfg.channel_dim}\n" +
                    f"num_blocks={cfg.num_blocks}\n" +
                    f"num_heads={cfg.num_heads}\n" +
                    f"mlp_ratio={cfg.mlp_ratio}\n" +
                    f"num_clusters={cfg.num_clusters}\n" +
                    f"num_projection_heads={cfg.num_projection_heads}\n" +
                    f"num_latent_blocks={cfg.num_latent_blocks}\n" +
                    f"if_latent_mlp={cfg.if_latent_mlp}\n" +
                    f"if_pointwise_mlp={cfg.if_pointwise_mlp}\n" +
                    f"cluster_head_mixing={cfg.cluster_head_mixing}"
                )
            model = bench.ClusterAttentionTransformer(
                in_dim=c_in,
                out_dim=c_out,
                channel_dim=cfg.channel_dim,
                num_blocks=cfg.num_blocks,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                num_clusters=cfg.num_clusters,
                num_projection_heads=cfg.num_projection_heads,
                num_latent_blocks=cfg.num_latent_blocks,
                if_latent_mlp=cfg.if_latent_mlp,
                if_pointwise_mlp=cfg.if_pointwise_mlp,
                cluster_head_mixing=cfg.cluster_head_mixing,
                act=cfg.act,
            )
        elif cfg.model_type == 2:
            if GLOBAL_RANK == 0:
                print(
                    f"Using SkinnyCAT with\n" +
                    f"channel_dim={cfg.channel_dim}\n" +
                    f"num_blocks={cfg.num_blocks}\n" +
                    f"num_clusters={cfg.num_clusters}\n" +
                    f"num_heads={cfg.num_heads}\n" +
                    f"cluster_head_mixing={cfg.cluster_head_mixing}"
                )
            model = bench.SkinnyCAT(
                in_dim=c_in,
                out_dim=c_out,
                channel_dim=cfg.channel_dim,
                num_blocks=cfg.num_blocks,
                num_clusters=cfg.num_clusters,
                num_heads=cfg.num_heads,
                act=cfg.act,
                cluster_head_mixing=cfg.cluster_head_mixing,
            )
        elif cfg.model_type == 9:
            if GLOBAL_RANK == 0:
                print(f"Using SparseTransformer.")
            model = bench.SparseTransformer(
                in_dim=c_in,
                out_dim=c_out,
                hidden_dim=cfg.channel_dim,
                num_blocks=cfg.num_blocks,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                num_slices=cfg.num_clusters,
                qk_norm=cfg.qk_norm,
                k_val=cfg.topk,
            )
        else:
            raise NotImplementedError("No unconditioned model selected.")
        
    # Use masked model for timeseries datasets
    if cfg.dataset in ['airfoil', 'cylinder_flow']:
        if GLOBAL_RANK == 0:
            print(f"Using masked model for timeseries datasets {cfg.dataset}")
        model = am.MaskedModel(model, mask=True, reduce_graph=False)

    if GLOBAL_RANK == 0:
        print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    #=================#
    # TRAIN
    #=================#

    callback = mlutils.Callback(case_dir)

    if cfg.model_type in [9,] and (time_cond == False):
        callback = bench.SparsityCallback(case_dir,)
    if cfg.dataset in ['airfoil', 'cylinder_flow']:
        callback = bench.TimeseriesCallback(case_dir, mesh=cfg.model_type in [-1,])
    elif cfg.dataset in ['elasticity', 'plasticity', 'darcy', 'airfoil_steady', 'pipe', 'navier_stokes',
                         'shapenet_car',]:
        callback = bench.RelL2Callback(case_dir, metadata['x_normalizer'], metadata['y_normalizer'])

    if cfg.train and cfg.epochs > 0:

        #----------#
        # batch_size
        #----------#

        _batch_size  = cfg.batch_size
        if cfg.dataset == 'airfoil':
            _batch_size = 1
            batch_size_ = _batch_size_ = 1 # 20
        elif cfg.dataset == 'cylinder_flow':
            batch_size_ = _batch_size_ = 1 # 50
        elif cfg.dataset in ['elasticity', 'plasticity', 'darcy', 'airfoil_steady', 'pipe', 'navier_stokes',
                             'shapenet_car',]:
            batch_size_ = _batch_size_ = 50

        #----------#
        # lossfun
        #----------#

        if cfg.dataset in ['elasticity', 'plasticity', 'darcy', 'airfoil_steady', 'pipe', 'navier_stokes',
                           'shapenet_car',]:
            if GLOBAL_RANK == 0:
                print(f"Using RelL2Loss for {cfg.dataset} dataset")
            lf = bench.RelL2Loss()
            def lossfun(yh, y):
                y_normalizer = metadata['y_normalizer'].to(y.device)
                yh = y_normalizer.decode(yh)
                y  = y_normalizer.decode(y)
                return lf(yh, y)
        else:
            if GLOBAL_RANK == 0:
                print(f"Using MSELoss for {cfg.dataset} dataset")
            lossfun = torch.nn.MSELoss()

        kw = dict(
            device=device, gnn_loader=cfg.dataset in ['airfoil', 'cylinder_flow'], stats_every=cfg.epochs//10,
            make_optimizer=bench.make_optimizer, weight_decay=cfg.weight_decay, epochs=cfg.epochs,
            _batch_size=_batch_size, batch_size_=batch_size_, _batch_size_=_batch_size_,
            lossfun=lossfun, clip_grad_norm=cfg.clip_grad_norm, adam_betas=(cfg.adam_beta1, cfg.adam_beta2),
        )

        #----------#
        # LR scheduler
        #----------#

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

        #----------#
        # Noise schedule
        #----------#
        if cfg.model_type in [9,]:
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
        if (cfg.model_type in [9,]) and cfg.dataset == 'elasticity':
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
            
        if cfg.dataset in ['darcy']:

            r = 5
            h = int(((421 - 1) / r) + 1)
            s = h
            dx = 1.0 / s
            lf = bench.RelL2Loss()

            def batch_lossfun(trainer, model, batch):
                x, y = batch
                yh = model(x)

                y_normalizer = metadata['y_normalizer'].to(y.device)
                yh = y_normalizer.decode(yh)
                y  = y_normalizer.decode(y)

                l2 = lf(yh, y)
                (gt_grad_x, gt_grad_y), (pred_grad_x, pred_grad_y) = bench.darcy_deriv_loss(yh, y, s, dx)
                deriv_loss = lf(pred_grad_x, gt_grad_x) + lf(pred_grad_y, gt_grad_y)

                loss = 0.1 * deriv_loss + l2
                return loss

            trainer.batch_lossfun = batch_lossfun

        if cfg.dataset in ['airfoil', 'cylinder_flow']:
            if GLOBAL_RANK == 0:
                print(f"Using masked loss for timeseries datasets {cfg.dataset}")
            batch_lossfun = am.MaskedLoss(mask=True)
            trainer.batch_lossfun = batch_lossfun

        #-------------#
        # load snapshot
        #-------------#

        if cfg.restart_file is not None:
            trainer.load(cfg.restart_file)

        #-------------#
        # train
        #-------------#

        trainer.train()

    #=================#
    # ANALYSIS
    #=================#
    
    if cfg.eval:
        if device != 'cpu' and device != torch.device('cpu'):
            torch.cuda.empty_cache()
        trainer = mlutils.Trainer(
            model, _data, data_, make_optimizer=bench.make_optimizer, device=device
        )
        trainer.make_dataloader()
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
    max_steps: int = 500
    init_step: int = 100
    init_case: int = 0
    exclude: bool = True
    train_rollout_noise: float = 0.

    # training arguments
    epochs: int = 100
    batch_size: int = 1
    weight_decay: float = 0e-0
    learning_rate: float = 1e-3
    schedule: str = 'OneCycleLR'
    one_cycle_pct_start:float = 0.10
    one_cycle_div_factor: float = 1e4
    one_cycle_final_div_factor: float = 1e4
    one_cycle_three_phase: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    clip_grad_norm: float = 1.0

    # model
    model_type: int = 0 # -1: MeshGraphNet, 0: Transolver, 1: CAT, 9: SparseTransformer
    act: str = None
    channel_dim: int = 64
    num_blocks: int = 8
    num_heads: int = 8
    mlp_ratio: float = 2.
    num_clusters: int = 64
    
    # CAT
    num_projection_heads: int = None
    num_latent_blocks: int = 1
    if_latent_mlp: bool = False
    if_pointwise_mlp: bool = True
    cluster_head_mixing: bool = True
    
    # Transolver
    conv2d: bool = False
    unified_pos: bool = False

    # Sparse Transformer
    topk: int = 0
    qk_norm: bool = True
    gamma_min: float = 0e-4
    gamma_init: float = 1e-2
    gamma_schedule: str = 'constant'
    noise_init: float = 1e-1
    noise_schedule: str = 'linear'

if __name__ == "__main__":
    
    DISTRIBUTED = mlutils.is_torchrun()
    GLOBAL_RANK = int(os.environ['RANK']) if DISTRIBUTED else 0
    WORLD_SIZE = int(os.environ['WORLD_SIZE']) if DISTRIBUTED else 1
    device = mlutils.select_device()

    #===============#
    cfg = CLI(Config, as_positional=False)
    #===============#
    
    if not (cfg.train or cfg.eval):
        print("No mode selected. Select one of train, eval")
        exit()

    #===============#
    mlutils.set_seed(cfg.seed)
    #===============#

    case_dir = os.path.join(CASEDIR, cfg.exp_name)

    if cfg.train and not cfg.eval:
        if cfg.dataset is None:
            print("No dataset selected.")
            exit()

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