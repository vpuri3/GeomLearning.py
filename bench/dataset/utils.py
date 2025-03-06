# Suppress TensorFlow warnings
import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, Subset

import bench

def load_tfrecord(path, meta):
    """Load and parse a TFRecord dataset.
    
    Args:
        path (str): Path to the TFRecord file
        meta (dict): Metadata dictionary containing field names and feature descriptions
        
    Returns:
        list: List of dictionaries containing the parsed data
    """

    import tensorflow as tf

    # Force CPU-only execution
    tf.config.set_visible_devices([], 'GPU')  # Hide all GPUs
    tf.config.set_visible_devices([tf.config.list_physical_devices('CPU')[0]], 'CPU')  # Restrict to CPU

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info+warn+error, 2=warn+error, 3=error only
    tf.get_logger().setLevel('ERROR')  # Suppress TF logging
    tf.autograph.set_verbosity(0)  # Suppress AutoGraph warnings

    ds = tf.data.TFRecordDataset(path)
    ds = ds.map(lambda x: tf.io.parse_single_example(x, {k: tf.io.VarLenFeature(tf.string) for k in meta['field_names']}))
    data = []
    for example in ds:
        sample = {}
        for key, field in meta['features'].items():
            raw_data = tf.io.decode_raw(example[key].values, getattr(tf, field['dtype']))
            sample[key] = tf.reshape(raw_data, field['shape']).numpy()
        data.append(sample)
    return data

def convert_to_pyg_data(raw_data):
    """Convert raw MeshGraphNets data to PyTorch Geometric Data objects."""
    import torch_geometric as pyg
    pyg_data_list = []
    
    for (i, graph) in enumerate(raw_data):
        # Static features (same across timesteps)
        cells = torch.tensor(graph["cells"][0], dtype=torch.long)
        pos = torch.tensor(graph["mesh_pos"][0], dtype=torch.float)
        node_type = torch.tensor(graph["node_type"][0], dtype=torch.long).squeeze(-1)
        node_type_onehot = torch.nn.functional.one_hot(node_type, num_classes=7).float()
        
        # Create edges from triangular cells
        edge_index = []
        for triangle in cells:
            edge_index.extend([
                [triangle[0], triangle[1]],
                [triangle[1], triangle[0]],
                [triangle[1], triangle[2]],
                [triangle[2], triangle[1]],
                [triangle[2], triangle[0]],
                [triangle[0], triangle[2]],
            ])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # edge features
        dx = pos[edge_index[0], 0] - pos[edge_index[1], 0]
        dy = pos[edge_index[0], 1] - pos[edge_index[1], 1]
        edge_attr = torch.stack([dx, dy], dim=-1)

        # Get full velocity sequence [n_steps, n_nodes, 2]
        velocity = torch.tensor(graph["velocity"], dtype=torch.float)
        n_steps = velocity.shape[0]
        
        # Create training pairs for autoregressive rollout
        for t in range(n_steps - 1):
            x = torch.cat([pos, velocity[t], node_type_onehot,], dim=-1)
            y = velocity[t + 1]
            
            if "pressure" in graph:
                x = torch.cat([x, torch.tensor(graph["pressure"][t], dtype=torch.float)], dim=-1)
                y = torch.cat([y, torch.tensor(graph["pressure"][t + 1], dtype=torch.float)], dim=-1)
            if "density" in graph:
                x = torch.cat([x, torch.tensor(graph["density"][t], dtype=torch.float)], dim=-1)
                y = torch.cat([y, torch.tensor(graph["density"][t + 1], dtype=torch.float)], dim=-1)

            data = pyg.data.Data(
                x=x, y=y, edge_attr=edge_attr, edge_index=edge_index,
                pos=pos, cells=cells, node_type=node_type,
                case_name=f'case_{i}', time_step=t,
            )
            pyg_data_list.append(data)
    
    return pyg_data_list


def load_dataset(dataset_name, DATADIR_BASE):
    """Load a dataset by name.
    
    Args:
        dataset_name (str): Name of the dataset to load ('elasticity', 'airfoil', or 'cylinder_flow')
        
    Returns:
        tuple: (train_data, test_data) containing the loaded datasets
    """
    if dataset_name == 'elasticity':
        DATADIR = os.path.join(DATADIR_BASE, 'GeoFNO', 'elasticity')
        PATH_Sigma = os.path.join(DATADIR, 'Meshes', 'Random_UnitCell_sigma_10.npy')
        PATH_XY = os.path.join(DATADIR, 'Meshes', 'Random_UnitCell_XY_10.npy')

        input_s = np.load(PATH_Sigma)
        input_s = torch.tensor(input_s, dtype=torch.float).permute(1, 0).unsqueeze(-1)
        input_xy = np.load(PATH_XY)
        input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2, 0, 1)
        
        ntrain = 1000
        ntest = 200
        
        y_normalizer = bench.UnitTransformer(input_s[:ntrain])
        input_s = y_normalizer.encode(input_s)

        dataset = TensorDataset(input_xy, input_s)
        train_data = Subset(dataset, range(ntrain))
        test_data = Subset(dataset, range(len(dataset)-ntest, len(dataset)))
        
        return train_data, test_data
        
    elif dataset_name in ['airfoil', 'cylinder_flow']:
        DATADIR = os.path.join(DATADIR_BASE, 'MeshGraphNets', dataset_name)

        # Load metadata
        with open(os.path.join(DATADIR, 'meta.json'), 'r') as fp:
            meta = json.loads(fp.read())
            
        # Load train and test data
        train_data = load_tfrecord(os.path.join(DATADIR, 'train.tfrecord'), meta)
        test_data = load_tfrecord(os.path.join(DATADIR, 'test.tfrecord'), meta)

        train_data = convert_to_pyg_data(train_data)
        test_data  = convert_to_pyg_data(test_data)
        
        return train_data, test_data
        
    else:
        raise ValueError(f"Dataset {dataset_name} not found.") 