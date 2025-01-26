import train
import os
import torch
import argparse
import numpy as np
from torch_geometric.data import Data
from normalization import *

from models.Transolver import Model

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../dataset_f6')
parser.add_argument('--save_dir', default='/data/PDE_data/mlcfd_data/preprocessed_data')
parser.add_argument('--fold_id', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--val_iter', default=10, type=int)
parser.add_argument('--cfd_config_dir', default='cfd/cfd_params.yaml')
parser.add_argument('--model',default = 'Transolver')
parser.add_argument('--cfd_mesh', action='store_true')
parser.add_argument('--r', default=0.2, type=float)
parser.add_argument('--weight', default=0.5, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batch_size', default=1, type=float)
parser.add_argument('--nb_epochs', default=200, type=float)  #res_1 = 1000, others = 200
parser.add_argument('--preprocessed', default=1, type=int)
args = parser.parse_args()
print(args)

hparams = {'lr': args.lr, 'batch_size': args.batch_size, 'nb_epochs': args.nb_epochs}

n_gpu = torch.cuda.device_count()
use_cuda = 0 <= args.gpu < n_gpu and torch.cuda.is_available()
device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')

# train_data, val_data, coef_norm = load_train_val_fold(args, preprocessed=args.preprocessed)
# train_ds = GraphDataset(train_data, use_cfd_mesh=args.cfd_mesh, r=args.r)
# val_ds = GraphDataset(val_data, use_cfd_mesh=args.cfd_mesh, r=args.r)

# file_path= os.path.join(args.data_dir, 'processed_set.pt')
# dataset = torch.load(file_path)[0:1000]
# train_ds = dataset[0:800]
# test_dataset = dataset[800:1000]

def cube_elems_to_edges(elems):
    N, _ = np.shape(elems)
    # Assuming node indexing as in: https://www.strand7.com/strand7r3help/Content/Topics/Elements/ElementsBricksElementTypes.htm
    edge_A = np.array([1,2,3,4,5,6,7,8,1,4,2,3])-1
    edge_B = np.array([2,3,4,1,6,7,8,5,5,8,6,7])-1
    edges = np.zeros([2,N*12])
    for i in range(N):
        new_idx = np.arange(i*12, (i+1)*12)
        edges[0, new_idx] = elems[i, edge_A]
        edges[1, new_idx] = elems[i, edge_B]

    edges = np.sort(edges, axis=0)
    edges = np.unique(edges, axis=1)

    edges = np.concatenate([edges, np.flipud(edges)], axis=1)
    return edges

def get_displacement_graph(name):

    results = np.load(name)
    verts, elems, disp, von_mises = results["verts"], results["elems"], results["disp"], results["von_mises_stress"]

    edges = cube_elems_to_edges(elems) - 1

    x = torch.tensor(verts, dtype=torch.float)
    y = torch.tensor(von_mises.flatten()/1000.0, dtype=torch.float).reshape(-1,1)
    edge_index = torch.tensor(edges, dtype=torch.long) 
    # print(torch.max(edge_index))

    data = Data(x=x, edge_index=edge_index, y=y)
    return data

def get_max_displacement_in_folder(folder):
    files = sorted(os.listdir(folder))
    maxes = []
    bad_idx = []
    for i, file in enumerate(files):
        name = os.path.join(folder, file)
        data = np.load(name)
        maxval = np.max(np.abs(data["disp"][:,2]))
        maxes.append(maxval)
        print(i, end="\r")
        if maxval > 1:
            bad_idx.append(i)
    maxes = np.array(maxes)
    return maxes, bad_idx


train_path = '../data_train'
train_files = sorted(os.listdir(train_path))
    
maxes, bad = get_max_displacement_in_folder("../data_train")
print('bad train: ',bad)
bad_names = [train_files[i] for i in bad]
for name in bad_names:
    train_files.remove(name)

train_dataset = []
for idx in range(len(train_files)):
    X = get_displacement_graph(os.path.join(train_path,train_files[idx]))
    train_dataset.append(X)

#Normalization
mean_vec_x,std_vec_x,mean_vec_y,std_vec_y = get_stats(train_dataset)
train_dataset = [normalize_all(train_dataset[i],mean_vec_x,std_vec_x,mean_vec_y,std_vec_y) for i in range(len(train_dataset))]

val_path = '../data_val'
val_files = sorted(os.listdir(val_path))

maxes, bad_te = get_max_displacement_in_folder("../data_val")
print('bad test: ',bad_te)
bad_names_te = [val_files[i] for i in bad_te]
for name in bad_names_te:
    val_files.remove(name)

val_dataset = []
for idx in range(len(val_files)):
    X = get_displacement_graph(os.path.join(val_path,val_files[idx]))
    val_dataset.append(X)
#Normalization
val_dataset = [normalize_all(val_dataset[i],mean_vec_x,std_vec_x,mean_vec_y,std_vec_y) for i in range(len(val_dataset))]

if args.model == 'Transolver':
    # model = Model(n_hidden=256, n_layers=8, space_dim=3,
    #               fun_dim=0,
    #               n_head=8,
    #               mlp_ratio=2, out_dim=1,
    #               slice_num=32,
    #               unified_pos=0).cuda()
    # model = Model(n_hidden=128, n_layers=7, space_dim=3,
    #             fun_dim=0,
    #             n_head=8,
    #             mlp_ratio=1, out_dim=1,
    #             slice_num=64,
    #             unified_pos=0).cuda() #num params 628985
    model = Model(n_hidden=128, n_layers=6, space_dim=3,
            fun_dim=0,
            n_head=8,
            mlp_ratio=1, out_dim=1,
            slice_num=32,
            unified_pos=0).cuda()

path = f'metrics/res_full_epochs_200'
if not os.path.exists(path):
    os.makedirs(path)

model = train.main(device, train_dataset, val_dataset, model, hparams, path)
