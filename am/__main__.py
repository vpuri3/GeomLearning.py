# 3rd party
import torch
import numpy as np
import torch_geometric as pyg
from tqdm import tqdm

# builtin
import os
import shutil
import argparse

# local
import am
import mlutils

DATADIR_RAW = "/home/shared/netfabb_ti64_hires_raw/"
DATADIR_TIMESERIES = "/home/shared/netfabb_ti64_hires_timeseries/"
DATADIR_FINALTIME  = "/home/shared/netfabb_ti64_hires_finaltime/"

def train_loop(model, data, E=100, lrs=None, nepochs=None, **kw):
    if lrs is None:
        lrs = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6]
    if nepochs is None:
        nepochs = [.1*E, .2*E, .25*E, .25*E, .1*E, .1*E]
        nepochs = [int(e) for e in nepochs]
    assert len(lrs) == len(nepochs)

    for i in range(len(lrs)):
        kwargs = dict(
            **kw, lr=lrs[i], nepochs=nepochs[i], print_config=(i==0),
        )
        trainer = mlutils.Trainer(model, data, **kwargs)
        trainer.train()

    return model

def train_finaltime(device, outdir, resdir, train=True):
    outname = os.path.join(outdir, "gnn")
    resname = os.path.join(resdir, "gnn")

    modelfile  = outname + ".pth"
    imagefile1 = resname + ".png"

    vis_dir = os.path.join(resdir, 'vis_finaltime')
    os.makedirs(vis_dir, exist_ok=True)

    #=================#
    # DATA: only consider first 100 cases
    #=================#
    DATADIR = os.path.join(DATADIR_FINALTIME, r"data_0-100")
    dataset = am.GraphDataset(DATADIR)

    #=================#
    # MODEL
    #=================#
    ci, ce, co, w, num_layers = 3, 3, 1, 64, 4
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

    for i in tqdm(range(20)):
        graph = dataset[i]
        fig = am.visualize_mpl(graph.x, graph.y, graph.edge_index)
        fig.savefig(os.path.join(vis_dir, f'data{str(i).zfill(2)}'), dpi=300)
    
        mesh = am.mesh_pyv(graph.x, graph.elems)
        mesh.point_data['target'] = graph.y.numpy(force=True)
        mesh.save(os.path.join(vis_dir, f'data{str(i).zfill(2)}.vtu'))

    return

def test_timeseries_data_extraction():
    ext_dir = "/home/shared/netfabb_ti64_hires_out/extracted/SandBox/"
    out_dir = "/home/shared/netfabb_ti64_hires_out/tmp/"
    errfile = os.path.join(out_dir, "error.txt")

    # consider a single case
    case_dir = os.path.join(ext_dir, "101635_11b839a3_5")
    # case_dir = os.path.join(ext_dir, "83419_82b6bccd_0")
    # case_dir = os.path.join(ext_dir, "77980_f6ed5970_4")

    # info = am.get_case_info(case_dir)
    # results = am.get_timeseries_results(case_dir)

    am.extract_from_dir(ext_dir, out_dir, errfile, timeseries=True)

    return

def view_timeseries_data(resdir):
    DATADIR = os.path.join(DATADIR_TIMESERIES, r"data_0-100")
    case_files = [f for f in os.listdir(DATADIR) if f.endswith(".pt")]
    vis_dir = os.path.join(resdir, 'vis_timeseries')

    for i in tqdm(range(20)):
        case_file = case_files[i]
        case_path = os.path.join(DATADIR, case_file)
        case_data = am.timeseries_dataset(case_path)
        out_dir   = os.path.join(vis_dir, f'case{str(i).zfill(2)}')
        visualize_timeseries_pyv(case_data, out_dir)

    return

def visualize_timeseries_pyv(dataset, out_dir):
    N = len(dataset)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for i in range(N):
        graph = dataset[i]
        mesh = am.mesh_pyv(graph.x, graph.elems)
        mesh.point_data['target'] = graph.y.numpy(force=True)
        mesh.save(os.path.join(out_dir, f'data{str(i).zfill(2)}.vtu'))

    pvd_file = os.path.join(out_dir, 'time_series.pvd')
    am.write_pvd(pvd_file, N, 'data')

    return

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
    # train_finaltime(device, outdir, resdir, train=False)

    #===============#
    # Timeseries data
    #===============#
    # am.extract_zips(DATADIR_RAW, DATADIR_TIMESERIES, timeseries=True)
    view_timeseries_data(resdir)

    pass
#
