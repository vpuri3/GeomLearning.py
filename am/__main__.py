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
        lrs = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6]
    if nepochs is None:
        nepochs = [.05*E, .25*E, .25*E, .25*E, .1*E, .1*E]
        nepochs = [int(e) for e in nepochs]
    assert len(lrs) == len(nepochs)

    for i in range(len(lrs)):
        kwargs = dict(
            **kw, lr=lrs[i], nepochs=nepochs[i], print_config=(i==0),
        )
        trainer = mlutils.Trainer(model, data, **kwargs)
        trainer.train()

    return model

#======================================================================#
def train_timeseries(device, outdir, resdir, train=True):
    outname = os.path.join(outdir, "gnn")
    resname = os.path.join(resdir, "gnn")

    modelfile  = outname + ".pth"
    imagefile1 = resname + ".png"

    vis_dir = os.path.join(resdir, 'gnn_timeseries')
    os.makedirs(vis_dir, exist_ok=True)

    #=================#
    # DATA: only consider first 100 cases
    #=================#

    DATADIR = os.path.join(DATADIR_TIMESERIES, r"data_0-100")
    dataset = am.TimeseriesDataset(DATADIR, merge=True)
    case_names = [f[:-3] for f in os.listdir(DATADIR) if f.endswith(".pt")]

    case_num = 2
    case_name = case_names[case_num]
    idx_case  = dataset.case_range(case_name)
    case_data = dataset[idx_case]

    for graph in case_data:
        print(graph)
        break

    def makedata(graph):
        md = graph.metadata
        time_step = md['time_step']
        pos_norm  = mlutils.normalize(graph.pos , md['pos_bar' ], md['pos_std' ])
        disp_norm = mlutils.normalize(graph.disp, md['disp_bar'], md['disp_std'])

        edge_norm = graph.edge_dxyz / md['pos_std']

        graph.x = torch.cat([pos_norm ], dim=-1)
        graph.y = torch.cat([disp_norm], dim=-1)
        graph.edge_attr = edge_norm

        return graph

    assert False

    #=================#
    # MODEL
    #=================#
    ci, ce, co, w, num_layers = 3, 3, 3, 256, 4
    model = mlutils.MeshGraphNet(ci, ce, co, w, num_layers)

    #=================#
    # TRAIN
    #=================#
    if train:
        kw = dict(device=device, E=200, gnn=True, _batch_size=1,)
        train_loop(model, dataset, **kw)
        torch.save(model.to("cpu").state_dict(), modelfile)

    #=================#
    # VISUALIZE
    #=================#
    model.eval()
    model.load_state_dict(torch.load(modelfile, weights_only=True))

    # for i in range(0,5):
    #     # graph = dataset[i]
    #     # fig = am.visualize_mpl(graph, 'temp')
    #     # fig.savefig(os.path.join(vis_dir, f'data{str(i).zfill(2)}'), dpi=300)
    #
    #     # fig = am.verify_connectivity(graph)
    #     # plt.show(block=True)
    #
    #     # mesh = am.mesh_pyv(graph.pos, graph.elems)
    #     # mesh.point_data['target'] = graph.temp.numpy(force=True)
    #     # mesh.save(os.path.join(vis_dir, f'data{str(i).zfill(2)}.vtu'))

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
    # make features/ labels
    # pos: want to rescale to (-1, 1), not normalize to zero mean, unit var
    # temp: want T --> (T - 293K) / (Tmax - 293)
    def transform_fn(graph):
        md = graph.metadata
        pos_norm  = mlutils.normalize(graph.pos , md['pos_bar' ], md['pos_std' ])
        disp_norm = mlutils.normalize(graph.disp, md['disp_bar'], md['disp_std'])

        edge_norm = graph.edge_dxyz / md['pos_std']

        graph.x = torch.cat([pos_norm ], dim=-1)
        graph.y = torch.cat([disp_norm], dim=-1)
        graph.edge_attr = edge_norm

        return graph

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
        kw = dict(device=device, E=200, gnn=True, _batch_size=1,)
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
    parser = argparse.ArgumentParser(description = 'Sandbox')
    parser.add_argument('--gpu_device', default=0, help='GPU device', type=int)
    args = parser.parse_args()

    device = mlutils.select_device()
    if device == "cuda":
        device += f":{args.gpu_device}"

    print(f"using device {device}")

    outdir = "./out/am/"
    resdir = "./res/am/"

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

    pass
#
