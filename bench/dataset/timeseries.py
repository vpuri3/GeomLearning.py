import os
import json
import torch
import torch.multiprocessing as mp
import torch_geometric as pyg
from tqdm import tqdm
from typing import Union, List, Optional

__all__ = [
    'TimeseriesDataset',
]
#======================================================================#

class TimeseriesDataset(pyg.data.Dataset):
    def __init__(
        self, 
        DATADIR: str, 
        dataset_split: str,
        num_workers: Optional[int] = None,
        transform=None, 
        force_reload=False,
    ):
        """
        Create dataset of time-series data from TFRecord files.

        Arguments:
            DATADIR (str): Base directory containing the dataset
            dataset_split (str): Name of the dataset split ('train' or 'test')
            transform: PyTorch Geometric transforms to apply
            force_reload (bool): Whether to force reprocessing of data
        """

        self.num_workers = num_workers if num_workers is not None else mp.cpu_count() // 2

        self.DATADIR = DATADIR
        self.dataset_name = DATADIR.split('/')[-1]
        self.dataset_split = dataset_split
        
        assert os.path.exists(DATADIR)
        assert dataset_split in ['train', 'test']
        assert self.dataset_name in ['airfoil', 'cylinder_flow']
        
        # Load metadata
        meta_path = os.path.join(self.DATADIR, 'meta.json')
        if not os.path.exists(meta_path):
            raise ValueError(f"Metadata file {meta_path} does not exist")
            
        with open(meta_path, 'r') as fp:
            self.meta = json.loads(fp.read())
        self.trajectory_length = self.meta['trajectory_length']
        self.num_steps = self.trajectory_length - 1
        self.num_cases = None
        
        # Ensure processed directory exists
        os.makedirs(self.processed_dir, exist_ok=True)
        
        super().__init__(transform=transform, force_reload=force_reload)
        
        # load in num_cases
        self.num_cases = len([f for f in os.listdir(self.processed_dir) if f.startswith('case_')])
        assert self.num_cases > 0
        
    @property
    def raw_paths(self):
        return [os.path.join(self.DATADIR, f'{self.dataset_split}.tfrecord'),]

    @property
    def processed_dir(self):
        return os.path.join(self.DATADIR, f'processed_{self.dataset_split}')
        
    @property
    def processed_paths(self):
        # If we haven't loaded any metadata yet, we need a minimal list
        # that includes at least metadata.pt to trigger processing
        metadata_path = os.path.join(self.processed_dir, 'metadata.pt')
        if not os.path.exists(metadata_path):
            return [metadata_path]
        return [metadata_path] + [os.path.join(self.processed_dir, f'case_{icase}.pt') for icase in range(self.num_cases)]

    def process(self):
        print(f"Processing {self.raw_paths[0]}...")
        
        # Load TFRecord data
        raw_data = load_tfrecord(self.raw_paths[0], self.meta)
        assert self.trajectory_length == raw_data[0]['velocity'].shape[0]
        self.num_cases = len(raw_data)
        
        for icase in tqdm(range(self.num_cases), ncols=80):
            self.process_single(raw_data, icase)

        # save dummy metadata
        torch.save([], os.path.join(self.processed_dir, 'metadata.pt'))

        return
    
    def process_single(self, raw_data, icase):
        case = raw_data[icase]
        data = convert_to_pyg_data(case)
        torch.save(data, os.path.join(self.processed_dir, f'case_{icase}.pt'))
        return
    
    def len(self):
        return self.num_steps * self.num_cases
    
    def case_range(self, case: int):
        """Get the range of indices for a specific case."""
        i0 = case * self.num_steps
        i1 = i0 + self.num_steps
        return range(i0, i1)
    
    def get(self, idx):
        """Get a single graph by global index."""
        case_idx = idx // self.num_steps
        time_step = idx % self.num_steps
        
        case_file = os.path.join(self.processed_dir, f'case_{case_idx}.pt')
        case_data = torch.load(case_file, weights_only=False, mmap=True)
        
        return case_data[time_step]

#======================================================================#
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

#======================================================================#
def convert_to_pyg_data(graph_dict):
    """Convert raw MeshGraphNets data to PyTorch Geometric Data objects."""
    
    # Static features (same across timesteps)
    cells = torch.tensor(graph_dict['cells'][0], dtype=torch.long)
    pos = torch.tensor(graph_dict['mesh_pos'][0], dtype=torch.float)
    node_type = torch.tensor(graph_dict['node_type'][0], dtype=torch.long).squeeze(-1)
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
    velocity = torch.tensor(graph_dict['velocity'], dtype=torch.float)
    n_steps = velocity.shape[0]
    
    # Create training pairs for autoregressive rollout
    for t in range(n_steps - 1):
        x = torch.cat([pos, velocity[t], node_type_onehot,], dim=-1)
        y = velocity[t + 1]
        
        if "pressure" in graph_dict:
            x = torch.cat([x, torch.tensor(graph_dict['pressure'][t], dtype=torch.float)], dim=-1)
            y = torch.cat([y, torch.tensor(graph_dict['pressure'][t + 1], dtype=torch.float)], dim=-1)
        if "density" in graph_dict:
            x = torch.cat([x, torch.tensor(graph_dict['density'][t], dtype=torch.float)], dim=-1)
            y = torch.cat([y, torch.tensor(graph_dict['density'][t + 1], dtype=torch.float)], dim=-1)

        data = pyg.data.Data(
            x=x, y=y, edge_attr=edge_attr, edge_index=edge_index,
            pos=pos, cells=cells, node_type=node_type,
            time_step=t,
        )
    
    return data

#======================================================================#
#