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

# DATADIR_BASE       = 'data/'
DATADIR_BASE       = '/home/shared/'
DATADIR_RAW        = os.path.join(DATADIR_BASE, 'netfabb_ti64_hires_raw')
DATADIR_TIMESERIES = os.path.join(DATADIR_BASE, 'netfabb_ti64_hires_timeseries')
DATADIR_FINALTIME  = os.path.join(DATADIR_BASE, 'netfabb_ti64_hires_finaltime')

#======================================================================#
def train_loop(model, data, E=100, lrs=None, nepochs=None, **kw):
    if lrs is None:
        lrs = [5e-4, 1e-4, 5e-5, 1e-5]
    if nepochs is None:
        nepochs = [.25*E, .25*E, .25*E, 0.25*E]

    nepochs = [int(e) for e in nepochs]
    assert len(lrs) == len(nepochs)
    for i in range(len(lrs)):
        kwargs = dict(
            **kw, lr=lrs[i], nepochs=nepochs[i], print_config=False,#(i==0),
        )
        trainer = mlutils.Trainer(model, data, **kwargs)
        trainer.train()

    return model

#======================================================================#
from torch import nn

class MaskedMGN(nn.Module):
    def __init__(self, ci, ce, co, w, num_layers, apply_mask=True):
        super().__init__()
        self.apply_mask = apply_mask
        self.gnn = mlutils.MeshGraphNet(ci, ce, co, w, num_layers)

    def reduce_graph(self, data):
        return data
        # # remove edges corresponding to nodes imask==False
        # mask  = data.mask
        # imask = torch.where(mask > 1e-4)
        # x = data.x
        # edge_index = data.edge_index
        # edge_attr = data.edge_attr

        return pyg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def forward(self, data, tol=1e-6):
        graph = self.reduce_graph(data)
        x = self.gnn(graph)

        if self.apply_mask:
            mask = data.mask.view(-1, 1)
            x = x * mask

        last_step_mask = (data.t <= 1. - tol).view(-1, 1)
        x = x * last_step_mask

        return x

#======================================================================#
class MergedTimeseriesTransform:
    def __init__(self):
        return

    def __call__(self, graph, tol=1e-4):
        N  = graph.pos.size(0)
        md = graph.metadata
        istep  = md['time_step']
        nsteps = md['time_steps']
        last_step = (istep + 1) == nsteps

        # mask
        if not last_step:
            zmax = md['zmax'][istep+1]
            mask = graph.pos[:,2] <= (zmax + tol)
        else:
            zmax = md['zmax'][-1]
            mask = torch.full((N,), True)

        # position
        # dataset: x, y [-30, 30], z [-25, 60] ([-25, 0] build plate)

        pos_shift = torch.tensor([ 0.,  1.,  0.])
        pos_scale = torch.tensor([30., 30., 60.])
        pos_norm = mlutils.normalize(graph.pos , pos_shift, pos_scale)

        # edges
        edge_norm = graph.edge_dxyz / pos_scale

        # time
        t  = torch.full((N, 1), graph.metadata['t_val'])
        dt = torch.full((N, 1), graph.metadata['dt_val'])

        # target fields
        if not last_step:
            disp_norm = graph.disp

            # disp_min = torch.tensor([-1., -1., -1.])
            # disp_max = torch.tensor([ 1.,  1.,  1.])
            # disp_shift, disp_scale = mlutils.shift_scale(graph.disp, disp_min, disp_max)
            # disp_norm = mlutils.normalize(graph.disp, disp_shift, disp_scale)

            disp_z = disp_norm[:, :, 2].unsqueeze(-1)

            disp_z_in  = disp_z[istep]
            disp_z_out = disp_z[istep+1]
            disp_z_out = (disp_z[istep+1] - disp_z[istep]) #/ md['dt_val']

        else:
            disp_z_in  = torch.zeros((N, 1))
            disp_z_out = torch.zeros((N, 1))

        # fields
        x = torch.cat([pos_norm, t, dt, disp_z_in,], dim=-1)
        y = torch.cat([disp_z_out,], dim=-1)
        edge_attr = edge_norm

        # assign to graph
        graph.x = x
        graph.y = y
        graph.t = t
        graph.mask = mask
        graph.edge_attr = edge_attr

        return graph

#======================================================================#
def train_timeseries(device, outdir, resdir, train=True):
    DISTRIBUTED = mlutils.is_torchrun()
    LOCAL_RANK = int(os.environ['LOCAL_RANK']) if DISTRIBUTED else 0

    outname = os.path.join(outdir, "gnn")
    resname = os.path.join(resdir, "gnn")

    modelfile  = outname + ".pth"
    imagefile1 = resname + ".png"

    vis_dir = os.path.join(resdir, 'gnn_timeseries')
    os.makedirs(vis_dir, exist_ok=True)

    #=================#
    # DATA: only consider first 100 cases
    #=================#

    transform = MergedTimeseriesTransform()
    DATADIR = os.path.join(DATADIR_TIMESERIES, r"data_0-100")
    dataset = am.TimeseriesDataset(DATADIR, merge=True, transform=transform)
    case_names = [f[:-3] for f in os.listdir(DATADIR) if f.endswith(".pt")]

    # just one case for now
    case_num = 2
    case_name = case_names[case_num]
    idx_case  = dataset.case_range(case_name)
    case_data = dataset[idx_case]
    dataset = case_data

    #=================#
    # MODEL
    #=================#
    ci, ce, co, w, num_layers = 6, 3, 1, 256, 5
    model = MaskedMGN(ci, ce, co, w, num_layers)

    #=================#
    # TRAIN
    #=================#
    if train:
        kw = dict(device=device, GNN=True, stats_every=10,
            E=200, _batch_size=1, weight_decay=0e-5,
        )
        train_loop(model, dataset, **kw)
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
        # Next Step prediction
        ###

        # auto_regressive = True
        auto_regressive = False

        with torch.no_grad():
            eval_data = []
            for data in case_data:
                data = transform(data)
                data.e = torch.zeros_like(data.y)
                eval_data.append(transform(data))

            K = 1
            for k in range(K, len(eval_data)):
                data  = eval_data[k].to(device)
                _data = eval_data[k-1].to(device)
                assert data.metadata['time_step'] == k

                if auto_regressive:
                    _data = _data.clone()
                    _data.x[:, -1] = _data.y[:, -1]

                data.y = model(_data) + data.disp[k-1, :, 2].unsqueeze(-1)
                data.e = data.y - data.disp[k, :, 2].unsqueeze(-1)

                l1 = nn.L1Loss()( data.e, 0 * data.e).item()
                l2 = nn.MSELoss()(data.e, 0 * data.e).item()
                r2 = mlutils.r2(data.y, data.disp[k,:,2])
                print(f'Step {k}: {l1, l2, r2}')

        out_dir = os.path.join(resdir, f'case{case_num}')
        am.visualize_timeseries_pyv(eval_data, out_dir, case_num, merge=True)

    return

#======================================================================#
def train_finaltime(device, outdir, resdir, train=True):
    outname = os.path.join(outdir, "gnn")
    resname = os.path.join(resdir, "gnn")

    modelfile  = outname + ".pth"
    imagefile1 = resname + ".png"

    vis_dir = os.path.join(resdir, 'gnn_finaltime')
    os.makedirs(vis_dir, exist_ok=True)

    #=================#
    # DATA: only consider first 100 cases
    #=================#
    def transform_fn(graph):
        md = graph.metadata
        pos_norm  = mlutils.normalize(graph.pos , md['pos' ][0], md['pos' ][1])
        disp_norm = mlutils.normalize(graph.disp, md['disp'][0], md['disp'][1])
        edge_norm = graph.edge_dxyz / md['pos'][1]

        x = torch.cat([pos_norm ], dim=-1)
        y = torch.cat([disp_norm], dim=-1)
        edge_attr = edge_norm

        return pyg.data.Data(x=x, y=y, edge_attr=edge_attr)

    DATADIR = os.path.join(DATADIR_FINALTIME, r"data_0-100")
    dataset = am.FinaltimeDataset(DATADIR, transform=transform_fn)

    #=================#
    # MODEL
    #=================#
    ci, ce, co, w, num_layers = 3, 3, 3, 64, 4
    model = mlutils.MeshGraphNet(ci, ce, co, w, num_layers)

    #=================#
    # TRAIN
    #=================#
    if train:
        kw = dict(device=device, E=200, GNN=True, _batch_size=1,)
        train_loop(model, dataset, **kw)
        torch.save(model.to("cpu").state_dict(), modelfile)

    #=================#
    # VISUALIZE
    #=================#
    model.eval()
    model.load_state_dict(torch.load(modelfile, weights_only=True))

    import matplotlib.pyplot as plt

    for i in range(0,5):
        pass
        # graph = dataset[i]
        # fig = am.visualize_mpl(graph, 'temp')
        # fig.savefig(os.path.join(vis_dir, f'data{str(i).zfill(2)}'), dpi=300)

        # fig = am.verify_connectivity(graph)
        # plt.show(block=True)

        # mesh = am.mesh_pyv(graph.pos, graph.elems)
        # mesh.point_data['target'] = graph.temp.numpy(force=True)
        # mesh.save(os.path.join(vis_dir, f'data{str(i).zfill(2)}.vtu'))

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
    parser.add_argument('--gpu_device', default=0, help='GPU device', type=int)
    args = parser.parse_args()

    DISTRIBUTED = mlutils.is_torchrun()
    LOCAL_RANK = int(os.environ['LOCAL_RANK']) if DISTRIBUTED else 0

    if DISTRIBUTED:
        mlutils.dist_setup()
        device = LOCAL_RANK
    else:
        device = mlutils.select_device()
        if device == "cuda":
            device += f":{args.gpu_device}"
        print(f"using device {device}")

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
    # am.extract_zips(DATADIR_RAW, DATADIR_TIMESERIES, timeseries=True)
    # vis_timeseries(resdir, merge=True)
    train_timeseries(device, outdir, resdir, train=True)

    #===============#
    if DISTRIBUTED:
        mlutils.dist_finalize()

    pass
#
