# 3rd party
import torch

import os
import yaml
import shutil
from jsonargparse import ArgumentParser, CLI
from dataclasses import dataclass

# local
import am
import mlutils

DATADIR_BASE       = 'data/'
# DATADIR_BASE       = '/home/shared/'
DATADIR_RAW        = os.path.join(DATADIR_BASE, 'netfabb_ti64_hires_raw')
DATADIR_TIMESERIES = os.path.join(DATADIR_BASE, 'netfabb_ti64_hires_timeseries')
DATADIR_FINALTIME  = os.path.join(DATADIR_BASE, 'netfabb_ti64_hires_finaltime')

CASEDIR = os.path.join('.', 'out', 'am')

#======================================================================#
def train_loop(model, _data, data_=None, E=100, lrs=None, nepochs=None, **kw):
    # if lrs is None:
    #     lrs = [5e-4, 1e-4, 5e-5, 1e-5]
    # if nepochs is None:
    #     nepochs = [.25*E, .25*E, .25*E, 0.25*E]

    if lrs is None:
        lrs = [5e-4, 1e-4,]
    if nepochs is None:
        nepochs = [.5*E, .5*E,]

    nepochs = [int(e) for e in nepochs]
    assert len(lrs) == len(nepochs)
    for i in range(len(lrs)):
        kwargs = dict(
            **kw, lr=lrs[i], nepochs=nepochs[i], print_config=False,#(i==0),
        )
        trainer = mlutils.Trainer(model, _data, data_, **kwargs)
        trainer.train()

    return model

#======================================================================#
def train_timeseries(cfg, device):
    DISTRIBUTED = mlutils.is_torchrun()
    LOCAL_RANK = int(os.environ['LOCAL_RANK']) if DISTRIBUTED else 0

    case_dir = os.path.join(CASEDIR, cfg.name)
    model_file = os.path.join(case_dir, 'model.pt')

    #=================#
    # DATA
    #=================#

    fields = dict(disp=cfg.disp, vmstr=cfg.vmstr, temp=cfg.temp)
    transform = am.MergedTimeseriesDataTransform(**fields)
    DATADIR = os.path.join(DATADIR_TIMESERIES, r"data_0-100")
    dataset = am.TimeseriesDataset(DATADIR, merge=True, transform=transform, num_workers=12)
    _data, data_ = am.split_timeseries_dataset(dataset, split=[0.8, 0.2])

    # # 3 (worst), 5, 6, 8, 9
    # _data, = am.split_timeseries_dataset(dataset, indices=[[3,5,6,8,9]])
    # data_ = None

    N = 20
    _data, data_ = am.split_timeseries_dataset(dataset, indices=[range(0,N), range(N,2*N)])

    #=================#
    # MODEL
    #=================#

    ci = 5 + cfg.disp + cfg.vmstr + cfg.temp
    ce = 3
    co = cfg.disp + cfg.vmstr + cfg.temp
    width = cfg.width
    num_layers = cfg.num_layers
    mask = cfg.mask
    blend = cfg.blend

    model = am.MaskedMGN(ci, ce, co, width, num_layers, mask=mask)

    #=================#
    # TRAIN
    #=================#

    if cfg.train:
        kw = dict(
            Opt='AdamW', device=device, GNN=True, stats_every=5,
            # _batch_size=1, batch_size_=4, _batch_size_=1, # baby expr
            _batch_size=4, batch_size_=12, _batch_size_=12, # v100-32
            # _batch_size=2, batch_size_=6, _batch_size_=6, # v100-16
            E=cfg.epochs, weight_decay=cfg.weight_decay,
        )

        train_loop(model, _data, data_, **kw)

        if LOCAL_RANK==0:
            print(f'Saving {model_file}')
            torch.save(model.to("cpu").state_dict(), model_file)

    #=================#
    # ANALYSIS
    #=================#

    if LOCAL_RANK == 0:

        print(f'Loading {model_file}')
        model_state = torch.load(model_file, weights_only=True, map_location='cpu')
        model.eval()
        model.load_state_dict(model_state)
        model.to(device)

        C = cfg.eval_cases
        K = cfg.autoreg_start

        _cases = [_data[_data.case_range(c)] for c in range(C)]
        cases_ = [data_[data_.case_range(c)] for c in range(C)] if data_ is not None else None

        for (cases, casetype) in zip([_cases, cases_], ['train', 'test']):
            if cases is None:
                print(f'No {casetype} data')
                continue
            for (i, case) in enumerate(cases):
                ii = str(i).zfill(2)
                case_data = cases[i]
        
                for (autoreg, ext) in zip([True, False], ['AR', 'NR']):
                    eval_data, l2s, r2s = am.march_case(
                        model, case_data, transform,
                        autoreg=autoreg, device=device, verbose=False,
                    )

                    out_dir = os.path.join(case_dir, f'{casetype}{ii}-{ext}')
                    am.visualize_timeseries_pyv(eval_data, out_dir, merge=True)

                    out_file = os.path.join(out_dir, 'stats.txt')
                    with open(out_file, 'w') as f:
                        f.write('Step\tMSE\tR-Square\n')
                        for (k, (l2,r2)) in enumerate(zip(l2s, r2s)):
                            f.write(f'{k}\t{l2s[k]}\t{r2s[k]}\n')

        # case_data = _data
        # eval_data, l2s, r2s = am.march_case(model, case_data, transform, autoreg=False, device=device)
        # # eval_data, l2s, r2s = am.march_case(model, case_data, transform, autoreg=True, K=5, device=device)
        #
        # out_dir = os.path.join(case_dir, f'case')
        # am.visualize_timeseries_pyv(eval_data, out_dir, merge=True)

    return

#======================================================================#
def train_finaltime(device, outdir, resdir, train=True):
    return

#======================================================================#
from tqdm import tqdm

def vis_timeseries(resdir, merge=None):
    DATADIR = os.path.join(DATADIR_TIMESERIES, r'data_0-100')
    dataset = am.TimeseriesDataset(DATADIR, merge=merge)

    vis_name = 'vis_timeseries_merged' if merge else 'vis_timeseries'
    vis_dir  = os.path.join(resdir, vis_name)
    case_names = [f[:-3] for f in os.listdir(DATADIR) if f.endswith(".pt")]

    for icase in tqdm(range(20)):
        case_name = case_names[icase]
        idx_case  = dataset.case_range(case_name)
        case_data = dataset[idx_case]
        out_dir   = os.path.join(vis_dir, f'case{str(icase).zfill(2)}')
        am.visualize_timeseries_pyv(case_data, out_dir, icase, merge=merge)

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

    # fields
    disp: bool  = True
    vmstr: bool = False
    temp: bool  = False

    # model
    width: int = 64
    num_layers: int = 5

    mask: bool = True
    blend: bool = True

    # training arguments
    epochs: int = 200
    weight_decay: float = 0e-4

    # eval arguments
    eval_cases: int = 10
    autoreg_start: int = 5

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

    if LOCAL_RANK == 0:
        case_dir = os.path.join(CASEDIR, cfg.name)

        if cfg.train:
            if os.path.exists(case_dir):
                ifrm = input(f'Remove case at {case_dir}? [Y/n]')
                if not 'n' in ifrm:
                    print(f'Removing {case_dir}')
                    shutil.rmtree(case_dir)
                else:
                    exit()

            os.makedirs(case_dir)
            config_file = os.path.join(case_dir, 'config.yaml')
            with open(config_file, 'w') as f:
                yaml.safe_dump(vars(cfg), f)

        else:
            assert os.path.exists(case_dir)

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
    # vis_timeseries(resdir, merge=True)
    train_timeseries(cfg, device)

    #===============#
    mlutils.dist_finalize()
    #===============#
    exit()
#
