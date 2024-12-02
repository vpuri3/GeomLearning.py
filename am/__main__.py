# 3rd party
import torch
import numpy as np
import torch_geometric as pyg
from tqdm import tqdm
import pyvista as pv

# builtin
import os
import shutil
import argparse

# local
import am
import mlutils

DATADIR_BASE       = 'data/'
# DATADIR_BASE       = '/home/shared/'
DATADIR_RAW        = os.path.join(DATADIR_BASE, 'netfabb_ti64_hires_raw')
DATADIR_TIMESERIES = os.path.join(DATADIR_BASE, 'netfabb_ti64_hires_timeseries')
DATADIR_FINALTIME  = os.path.join(DATADIR_BASE, 'netfabb_ti64_hires_finaltime')

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
class MergedTimeseriesProcessor:
    def __init__(self, disp=True, vmstr=True, temp=True):

        self.disp  = disp
        self.vmstr = vmstr
        self.temp  = temp

        # pos  : x, y [-30, 30] mm, z [-25, 60] mm ([-25, 0] build plate)
        # disp : x [-0.5, 0.5] mm, y [-0.05, 0.05] mm, z [-0.1, -1] mm
        # vmstr: [0, 5e3] Pascal (?)
        # temp : Celcius [25, 300]
        #
        # time: [0, 1]

        self.pos_scale = torch.tensor([30., 30., 60.])
        self.disp_scale  = 1.
        self.vmstr_scale = 1000.
        self.temp_scale  = 500. # TODO: adjust?

        return

    def __call__(self, graph, tol=1e-4):
        N  = graph.pos.size(0)
        md = graph.metadata
        istep  = md['time_step']
        nsteps = md['time_steps']
        last_step = (istep + 1) == nsteps

        # TODO:
        #    dz = zmax[istep+1] - zmax[istep]
        #
        # use dz to decide the interface width such that
        # interface fully encompasses one layer and ends at the next.
        # input to GNN should not have sharp discontinuity

        # mask
        if not last_step:
            zmax = md['zmax'][istep+1]
            mask = graph.pos[:,2] <= (zmax + tol)
        else:
            zmax = md['zmax'][-1]
            mask = torch.full((N,), True)

        # position
        pos = graph.pos / self.pos_scale

        # edges
        edge_dxyz = graph.edge_dxyz / self.pos_scale

        # time
        t  = torch.full((N, 1), graph.metadata['t_val'])
        dt = torch.full((N, 1), graph.metadata['dt_val'])

        # disp
        disp  = graph.disp  / self.disp_scale
        vmstr = graph.vmstr / self.vmstr_scale
        temp  = graph.temp  / self.temp_scale

        # target fields
        if not last_step:
            disp_z   = disp[:, :, 2].unsqueeze(-1)
            disp_in  = disp_z[istep]
            disp_out = (disp_z[istep+1] - disp_z[istep]) #/ md['dt_val']

            vmstr_in  = vmstr[istep]
            vmstr_out = (vmstr[istep+1] - vmstr[istep])

        else:
            disp_in  = torch.zeros((N, 1))
            disp_out = torch.zeros((N, 1))

            vmstr_in  = torch.zeros((N, 1))
            vmstr_out = torch.zeros((N, 1))

            temp_in  = torch.zeros((N, 1))
            temp_out = torch.zeros((N, 1))

        # features / labels
        xs = [pos, t, dt,]
        ys = []

        if self.disp:
            xs.append(disp_in)
            ys.append(disp_out)
        if self.vmstr:
            xs.append(vmstr_in)
            ys.append(vmstr_out)
        if self.temp:
            xs.append(temp_in)
            ys.append(temp_out)

        assert len(ys) > 0, f"At least one of disp, vmstr, temp must be True. Got {self.disp}, {self.vmstr}, {self.temp}."

        x = torch.cat(xs, dim=-1)
        y = torch.cat(ys, dim=-1)

        edge_attr = edge_dxyz

        return pyg.data.Data(
            x=x, y=y, t=t, mask=mask,
            edge_attr=edge_attr, edge_index=graph.edge_index,
            disp=graph.disp[istep], vmstr=graph.vmstr[istep], temp=graph.temp[istep],
        )

#======================================================================#
from torch import nn

@torch.no_grad()
def eval_case(model, case_data, fields, autoreg=True, K=1, verbose=True, device=None):

    disp  = fields['disp']
    vmstr = fields['vmstr']
    temp  = fields['temp']

    nfields = disp + vmstr + temp

    # TODO

    def makefields(data):
        xs = []
        if disp:
            xs.append(data.disp[:, 2].view(-1,1))
        if vmstr:
            xs.append(data.vmstr.view(-1,1))
        if temp:
            xs.append(data.temp.view(-1,1))

        return torch.cat(xs, dim=-1)

    if device is None:
        device = mlutils.select_device(device)

    model.to(device)

    eval_data = []
    for data in case_data:
        data = data.clone()
        data.y = data.disp[:, 2].unsqueeze(-1)
        data.e = torch.zeros_like(data.y)
        eval_data.append(data)

    for k in range(K, len(eval_data)):
        _data = eval_data[k-1].to(device) # given (k-1)-th step
        data  = eval_data[k  ].to(device) # predict k-th step

        if autoreg:
            _data = _data.clone()
            _data.x[:, -1] = _data.y[:, -1]

        data.y = model(_data) + _data.x[:, -1].unsqueeze(-1)
        data.e = data.y - data.disp[:, 2].unsqueeze(-1)

        if verbose:
            l1 = nn.L1Loss()( data.e, 0 * data.e).item()
            l2 = nn.MSELoss()(data.e, 0 * data.e).item()
            r2 = mlutils.r2(data.y, data.disp[:,2])
            print(f'Step {k}: {l1, l2, r2}')

    return eval_data

#======================================================================#
def train_timeseries(device, outdir, resdir, train=True):
    DISTRIBUTED = mlutils.is_torchrun()
    LOCAL_RANK = int(os.environ['LOCAL_RANK']) if DISTRIBUTED else 0

    outname = os.path.join(outdir, "gnn")
    resname = os.path.join(resdir, "gnn")
    modelfile  = outname + ".pt"

    vis_dir = os.path.join(resdir, 'gnn_timeseries')
    os.makedirs(vis_dir, exist_ok=True)

    #=================#
    # DATA
    #=================#

    # disp = True
    # vmstr = False
    # temp = False

    disp = False
    vmstr = True
    temp = False

    fields = dict(disp=disp, vmstr=vmstr, temp=temp)

    transform = MergedTimeseriesProcessor(**fields)
    DATADIR = os.path.join(DATADIR_TIMESERIES, r"data_0-100")
    dataset = am.TimeseriesDataset(DATADIR, merge=True, transform=transform, num_workers=12)

    _data, data_ = am.split_timeseries_dataset(dataset, [0.8, 0.2]) # RIGHT

    #=================#
    # MODEL
    #=================#

    ci = 5 + disp + vmstr + temp
    ce = 3
    co = disp + vmstr + temp
    width = 64
    num_layers = 5

    model = am.MaskedMGN(ci, ce, co, width, num_layers)

    #=================#
    # TRAIN
    #=================#

    if train:
        kw = dict(
            device=device, GNN=True, stats_every=5,
            _batch_size=4, batch_size_=12, _batch_size_=12,
            E=100, weight_decay=0e-5, Opt='AdamW',
        )

        train_loop(model, _data, data_, **kw)

        if LOCAL_RANK==0:
            torch.save(model.to("cpu").state_dict(), modelfile)

    #=================#
    # ANALYSIS
    #=================#

    if LOCAL_RANK == 0:

        model.eval()
        model_state = torch.load(modelfile, weights_only=True, map_location='cpu')
        model.load_state_dict(model_state)
        model.to(device)

        ###
        # choose case
        ###

        C = 5
        _cases = [_data[_data.case_range(c)] for c in range(C)]
        cases_ = [data_[data_.case_range(c)] for c in range(C)]

        case_data = _cases[0]
        # case_data = cases_[2]

        ###
        # Next Step prediction
        ###

        # eval_data = eval_case(model, case_data, fields, autoreg=False, device=device)
        # eval_data = eval_case(model, case_data, fields, autoreg=True, K=5, device=device)

        # out_dir = os.path.join(resdir, f'case{case_num}')
        # am.visualize_timeseries_pyv(eval_data, out_dir, case_num, merge=True)

    return

#======================================================================#
def train_finaltime(device, outdir, resdir, train=True):
    DISTRIBUTED = mlutils.is_torchrun()
    LOCAL_RANK = int(os.environ['LOCAL_RANK']) if DISTRIBUTED else 0

    outname = os.path.join(outdir, "gnn")
    resname = os.path.join(resdir, "gnn")
    modelfile  = outname + ".pt"

    vis_dir = os.path.join(resdir, 'gnn_timeseries')
    os.makedirs(vis_dir, exist_ok=True)

    #=================#
    # DATA
    #=================#

    disp = True
    vmstr = False

    transform = 0 # MergedTimeseriesProcessor(disp, vmstr)
    DATADIR = os.path.join(DATADIR_TIMESERIES, r"data_0-100")
    dataset = am.TimeseriesDataset(DATADIR, merge=True, transform=transform, num_workers=12)

    _data, data_ = am.split_timeseries_dataset(dataset, [0.8, 0.2]) # RIGHT

    #=================#
    # MODEL
    #=================#

    ci = 5 + disp + vmstr
    ce = 3
    co = disp + vmstr
    width = 64
    num_layers = 5

    model = am.MaskedMGN(ci, ce, co, width, num_layers)

    #=================#
    # TRAIN
    #=================#

    return

#======================================================================#
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
if __name__ == "__main__":

    mlutils.set_seed(123)
    parser = argparse.ArgumentParser(description = 'AM')
    args = parser.parse_args()

    DISTRIBUTED = mlutils.is_torchrun()
    LOCAL_RANK = int(os.environ['LOCAL_RANK']) if DISTRIBUTED else 0

    if DISTRIBUTED:
        mlutils.dist_setup()
        device = LOCAL_RANK
    else:
        device = mlutils.select_device()

    outdir = "./out/am/"
    resdir = "./res/am/"

    if LOCAL_RANK == 0:
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        if not os.path.exists(resdir):
            os.mkdir(resdir)

    #===============#
    # Final time data
    #===============#
    # am.extract_zips(DATADIR_RAW, DATADIR_FINALTIME)
    # train_finaltime(device, outdir, resdir, train=True)

    #===============#
    # Timeseries data
    #===============#
    # test_timeseries_extraction()
    # am.extract_zips(DATADIR_RAW, DATADIR_TIMESERIES, timeseries=True, num_workers=12)
    # vis_timeseries(resdir, merge=True)
    train_timeseries(device, outdir, resdir, train=True)

    #===============#
    if DISTRIBUTED:
        mlutils.dist_finalize()

    pass
#
