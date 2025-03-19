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
class TimeseriesDatasetTransform:
    def __init__(
        self,
        mesh=False, cells=False,
        orig=False, metadata=False,
        vel=True, pres=True, dens=True,
    ):
        self.mesh = mesh
        self.cells = cells
        self.orig  = orig
        self.metadata = metadata

        self.vel = vel
        self.pres = pres
        self.dens = dens
        self.nfields = 2 * vel + pres + dens

        self.pos_scale = torch.tensor([1., 1.])
        self.vel_scale  = torch.tensor([1., 1.])
        self.pres_scale  = torch.tensor([1.])
        self.dens_scale  = torch.tensor([1.])

        scale = []
        scale = [*scale, self.vel_scale ] if self.vel  else scale
        scale = [*scale, self.pres_scale] if self.pres else scale
        scale = [*scale, self.dens_scale ] if self.dens  else scale
        self.scale = torch.cat(scale, dim=-1)

        assert self.nfields == len(self.scale), f"{self.nfields} != {len(self.scale)}"

        return

    def normalize_fields(self, graph):
        pos   = graph.pos     / self.pos_scale
        vel  = graph.velocity / self.vel_scale
        pres = graph.pressure / self.pres_scale
        dens = graph.density  / self.dens_scale if hasattr(graph, 'density') else None
        edge_attr = graph.edge_attr / self.pos_scale
        return pos, vel, pres, dens, edge_attr
    
    def make_pyg_data(self, graph, edge_attr, **kw):
        data = pyg.data.Data(**kw)

        if self.mesh:
            data.edge_attr  = edge_attr
            data.edge_index = graph.edge_index
        if self.cells:
            data.cells = graph.cells
        if self.orig:
            data.pos = graph.pos
            data.velocity = graph.velocity
            data.pressure = graph.pressure
            data.density  = graph.density if hasattr(graph, 'density') else None
        if self.metadata:
            data.metadata = graph.metadata

        return data

    def makefields(self, data, time_step, scale=False):

        xs = []
        xs = [*xs, data.velocity[time_step]] if self.vel  else xs
        xs = [*xs, data.pressure[time_step]] if self.pres else xs
        xs = [*xs, data.density[time_step] ] if self.dens else xs

        out = torch.cat(xs, dim=-1)
        
        if scale:
            out = out / self.scale.to(xs[0].device)

        return out

    def __call__(self, graph):

        N  = graph.pos.size(0)
        md = graph.metadata
        time_step  = md['time_step'] # zero indexed
        time_steps = md['time_steps']
        last_step  = (time_step + 1) == time_steps
        
        # normalize fields
        pos, vel, pres, dens, edge_attr = self.normalize_fields(graph)

        # time
        t_val = md['dt'] * time_step
        T_val = md['dt'] * time_steps
        t = torch.full((N, 1), t_val)
        T = torch.full((N, 1), T_val)

        if last_step:

            vel_in  = torch.zeros((N, 2))
            vel_out = torch.zeros((N, 2))

            pres_in  = torch.zeros((N, 1))
            pres_out = torch.zeros((N, 1))

            dens_in  = torch.zeros((N, 1)) if dens is not None else None
            dens_out = torch.zeros((N, 1)) if dens is not None else None
        else:

            vel0  = vel[ time_step]
            pres0 = pres[time_step]
            dens0 = dens[time_step] if dens is not None else None

            vel1  = vel[ time_step + 1]
            pres1 = pres[time_step + 1]
            dens1 = dens[time_step + 1] if dens is not None else None

            vel_in  = vel0
            pres_in  = pres0
            dens_in  = dens0 if dens is not None else None

            vel_out  = vel1  - vel0
            pres_out = pres1 - pres0
            dens_out = dens1 - dens0 if dens is not None else None

        # features / labels
        xs = [pos, graph.node_type_onehot]
        ys = []

        if self.vel:
            xs.append(vel_in)
            ys.append(vel_out)
        if self.pres:
            xs.append(pres_in)
            ys.append(pres_out)
        if self.dens:
            xs.append(dens_in)
            ys.append(dens_out)

        x = torch.cat(xs, dim=-1)
        y = torch.cat(ys, dim=-1)

        assert y.size(-1) == self.nfields, f"At least one of vel, pres, dens must be True. Got {self.vel}, {self.pres}, {self.dens}."
        
        # make prediction only on NORMAL and OUTFLOW nodes
        # https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/common.py
        mask = torch.zeros(N, dtype=torch.bool)
        mask[graph.node_type == 0] = True
        mask[graph.node_type == 5] = True
        
        # Training noise? See MeshGraphNets paper

        data = self.make_pyg_data(
            graph,
            edge_attr,
            x=x, y=y,
            t=t, T=T,
            mask=mask
        )
        
        del graph

        return data

#======================================================================#

class TimeseriesDataset(pyg.data.Dataset):
    def __init__(
        self, 
        DATADIR: str, 
        dataset_split: str,
        transform=None, 
        force_reload=False,
        max_cases: int = None,
        max_steps: int = None,
    ):
        """
        Create dataset of time-series data from TFRecord files.

        Arguments:
            DATADIR (str): Base directory containing the dataset
            dataset_split (str): Name of the dataset split ('train' or 'test')
            transform: PyTorch Geometric transforms to apply
            force_reload (bool): Whether to force reprocessing of data
        """

        self.max_cases = max_cases
        self.max_steps = max_steps

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

        self.num_cases = None
        self.num_steps = self.trajectory_length
        
        # Ensure processed directory exists
        os.makedirs(self.processed_dir, exist_ok=True)
        
        super().__init__(transform=transform, force_reload=force_reload)
        
        # load in num_cases
        self.set_num_cases()
        assert self.num_cases > 0
        
        # set num_steps
        if self.max_steps is not None:
            print(f"Limiting {self.dataset_split} dataset to {self.max_steps} steps")
            self.num_steps = min(self.num_steps, self.max_steps)

    def set_num_cases(self):
        self.num_cases = len([f for f in os.listdir(self.processed_dir) if f.startswith('case_')])
        if self.max_cases is not None:
            print(f"Limiting {self.dataset_split} dataset to {self.max_cases} cases")
            self.num_cases = min(self.num_cases, self.max_cases)

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
        else:
            self.set_num_cases()
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
        i1 = i0 + self.num_steps - 1
        return range(i0, i1)
    
    def get(self, idx):
        case_idx = idx // self.num_steps
        time_step = idx % self.num_steps
        
        case_file = os.path.join(self.processed_dir, f'case_{case_idx}.pt')
        case_data = torch.load(case_file, weights_only=False, mmap=True)
        case_data.metadata = {
            'dt': self.meta['dt'],
            'time_steps': self.meta['trajectory_length'],
            'time_step': time_step,
        }
        
        return case_data

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

    velocity = torch.tensor(graph_dict['velocity'], dtype=torch.float) # [n_steps, n_nodes, 2]
    pressure = torch.tensor(graph_dict['pressure'], dtype=torch.float) # [n_steps, n_nodes, 1]
    if "density" in graph_dict:
        density = torch.tensor(graph_dict['density'], dtype=torch.float) # [n_steps, n_nodes, 1]
    else:
        density = None
        
    data = pyg.data.Data(
        cells=cells,
        pos=pos,
        node_type=node_type,
        node_type_onehot=node_type_onehot,
        edge_index=edge_index,
        edge_attr=edge_attr,
        velocity=velocity,
        pressure=pressure,
        density=density,
    )
    return data

#======================================================================#
#