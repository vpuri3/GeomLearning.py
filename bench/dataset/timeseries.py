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
        dataset_name: str,
        mesh=False, cells=False,
        orig=False, metadata=False,
        vel=True, pres=False, dens=False,
    ):
        self.dataset_name = dataset_name

        self.mesh = mesh
        self.cells = cells
        self.orig  = orig
        self.metadata = metadata

        self.vel = vel
        self.pres = pres
        self.dens = dens
        self.nfields = 2 * vel + pres + dens
            
        # normalization stats
        self.pos_min = torch.tensor([0., 0.])
        self.pos_max = torch.tensor([1., 1.])

        self.vel_shift = torch.tensor([0., 0.])
        self.vel_scale = torch.tensor([1., 1.])
        
        self.pres_shift = torch.tensor([0.])
        self.pres_scale = torch.tensor([1.])

        self.dens_shift = torch.tensor([0.])
        self.dens_scale = torch.tensor([1.])
    
        self.out_shift = torch.tensor([0.])
        self.out_scale = torch.tensor([1.])
        
    def apply_normalization_stats(self, norm_stats):
        self.pos_min = norm_stats['pos_min']
        self.pos_max = norm_stats['pos_max']

        self.vel_shift = norm_stats['vel_mean']
        self.vel_scale = norm_stats['vel_std']
        
        self.out_shift = norm_stats['output_mean'] * 0.
        self.out_scale = norm_stats['output_std']

    def make_fields(self, data, time_step):
        device = data.velocity.device

        xs = []
        if self.vel:
            vel = (data.velocity[time_step] - self.vel_shift.to(device)) / self.vel_scale.to(device)
            xs = [*xs, vel]
        if self.pres:
            pres = (data.pressure[time_step] - self.pres_shift.to(device)) / self.pres_scale.to(device)
            xs = [*xs, pres]
        if self.dens:
            dens = (data.density[time_step] - self.dens_shift.to(device)) / self.dens_scale.to(device)
            xs = [*xs, dens]

        out = torch.cat(xs, dim=-1)
        
        return out

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
            data.density  = graph.density if graph.get('density', None) is not None else None
        if self.metadata:
            data.metadata = graph.metadata

        return data

    def __call__(self, graph):

        N  = graph.pos.size(0)
        md = graph.metadata
        dt = md['dt']
        time_idx  = md['time_idx']  # {0, num_steps-1} == range(0, num_steps)
        init_step = md['init_step']
        num_steps = md['num_steps']
        last_step = (time_idx + 1) == num_steps
        time_step = time_idx + init_step
        
        t0 = dt * init_step
        tt = dt * (init_step + time_idx)
        tf = dt * (init_step + num_steps - 1)

        t_val = (tt - t0) / (tf - t0)
        T_val = (tf - t0) / (tf - t0)
        
        t = torch.full((N, 1), t_val)
        T = torch.full((N, 1), T_val)
        
        # get fields
        pos = (graph.pos - self.pos_min) / (self.pos_max - self.pos_min)
        edge_attr = graph.edge_attr / (self.pos_max - self.pos_min)

        # print(f"init_step: {init_step}, time_idx: {time_idx}, num_steps: {num_steps}, time_step: {time_step}, t: {t_val}, T: {T_val}, last_step: {last_step}")

        if last_step:
            vel_in  = torch.zeros((N, 2))
            vel_out = torch.zeros((N, 2))

            pres_in  = torch.zeros((N, 1))
            pres_out = torch.zeros((N, 1))

            dens_in  = torch.zeros((N, 1))
            dens_out = torch.zeros((N, 1))
        else:

            vel_in  = (graph.velocity[time_step] - self.vel_shift ) / self.vel_scale
            pres_in = (graph.pressure[time_step] - self.pres_shift) / self.pres_scale
            dens_in = (graph.density[ time_step] - self.dens_shift) / self.dens_scale if graph.get('density', None) is not None else None

            vel_out  = graph.velocity[time_step + 1] - graph.velocity[time_step]
            pres_out = graph.pressure[time_step + 1] - graph.pressure[time_step]
            dens_out = graph.density[ time_step + 1] - graph.density[ time_step] if graph.get('density', None) is not None else None
            
        # features / labels
        xs = [graph.node_type_onehot, pos,]
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
        
        y = (y - self.out_shift) / self.out_scale

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
        init_step: int = None,
        num_workers: int = None,
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
        self.num_workers = mp.cpu_count() // 2 if num_workers is None else num_workers
        
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
        self.init_step = init_step if init_step is not None else 0
        
        # Ensure processed directory exists
        os.makedirs(self.processed_dir, exist_ok=True)
        
        super().__init__(transform=transform, force_reload=force_reload)
        
        # load in num_cases
        self.set_num_cases()
        assert self.num_cases > 0
        
        # set num_steps
        if self.max_steps is not None:
            self.num_steps = min(self.num_steps, self.max_steps)

        # normalization stats
        norm_stats = self.compute_normalization_stats(verbose=False)
        self.transform.apply_normalization_stats(norm_stats)

    def set_num_cases(self):
        self.num_cases = len([f for f in os.listdir(self.processed_dir) if f.startswith('case_')])
        if self.max_cases is not None:
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
        i1 = i0 + self.num_steps
        return range(i0, i1)
    
    def get(self, idx):
        case_idx = idx // self.num_steps
        time_idx = idx %  self.num_steps
        
        case_file = os.path.join(self.processed_dir, f'case_{case_idx}.pt')
        case_data = torch.load(case_file, weights_only=False, mmap=True)
        case_data.metadata = {
            'dt': self.meta['dt'],
            'time_idx': time_idx,
            'init_step': self.init_step,
            'num_steps': self.num_steps,
        }
        
        return case_data

    def compute_normalization_stats(self, verbose=True):
        print(f"Computing normalization stats for {self.dataset_split} dataset...")

        norm_stats = {}
        orig = self.transform.orig
        self.transform.orig = True
        
        for graph in self:
            norm_stats = dict(pos_min = graph.pos.min(dim=0).values, pos_max = graph.pos.max(dim=0).values,)
            break

        stats = GraphNormStats(num_steps=self.num_steps, init_step=self.init_step)

        # mp.set_start_method('spawn', force=True)
        # with mp.Pool(self.num_workers) as pool:
        #     list(tqdm(
        #         pool.imap_unordered(stats.update, self), total=len(self),
        #         desc=f'Computing normalization for {self.dataset_split}',
        #         ncols=80,
        #     ))
        for graph in self:
            stats.update(graph)
        
        assert stats.num_graphs == len(self), f"Number of graphs processed {stats.num_graphs} != {len(self)}"
        norm_stats = dict(**norm_stats, **stats.compute())

        if verbose:
            print(f"pos_min: {norm_stats['pos_min']}")
            print(f"pos_max: {norm_stats['pos_max']}")

            print(f"vel_mean: {norm_stats['vel_mean']}")
            print(f"vel_std : {norm_stats['vel_std']}")

            print(f"input_mean: {norm_stats['input_mean']}")
            print(f"input_std : {norm_stats['input_std']}")

            print(f"output_mean: {norm_stats['output_mean']}")
            print(f"output_std : {norm_stats['output_std']}")

        self.transform.orig = orig

        return norm_stats

class GraphNormStats:
    def __init__(self, num_steps, init_step):
        self.num_graphs = 0
        self.num_steps  = num_steps
        self.init_step  = init_step
        self.idx_range  = range(init_step, init_step + num_steps)
        
        self.pos_min = torch.ones(2) *  torch.inf
        self.pos_max = torch.ones(2) * -torch.inf

        self.vel_mean    = 0.
        self.vel_std     = 0.
        self.input_mean  = 0.
        self.input_std   = 0.
        self.output_mean = 0.
        self.output_std  = 0.
        
    def update(self, graph):
        self.num_graphs  += 1
        self.vel_mean    += graph.velocity[self.idx_range].mean(dim=(0,1))
        self.vel_std     += graph.velocity[self.idx_range].std( dim=(0,1))
        self.input_mean  += graph.x.mean(dim=0)[7:]
        self.input_std   += graph.x.std(dim=0)[7:]
        self.output_mean += graph.y.mean(dim=0)
        self.output_std  += graph.y.std(dim=0)
        
        self.pos_min[0] = torch.min(self.pos_min[0], graph.pos[:,0].min())
        self.pos_min[1] = torch.min(self.pos_min[1], graph.pos[:,1].min())

        self.pos_max[0] = torch.max(self.pos_max[0], graph.pos[:,0].max())
        self.pos_max[1] = torch.max(self.pos_max[1], graph.pos[:,1].max())
        
        if graph.velocity[:self.num_steps].isnan().any():
            print(f"number of nans: {graph.velocity[:self.num_steps].isnan().sum()}")
            print(f"number of infs: {graph.velocity[:self.num_steps].isinf().sum()}")

        del graph
        
    def compute(self):
        assert self.num_graphs > 0
        return {
            # 'pos_min'    : self.pos_min,
            # 'pos_max'    : self.pos_max,
            'vel_mean'   : self.vel_mean    / self.num_graphs,
            'vel_std'    : self.vel_std     / self.num_graphs,
            'input_mean' : self.input_mean  / self.num_graphs,
            'input_std'  : self.input_std   / self.num_graphs,
            'output_mean': self.output_mean / self.num_graphs,
            'output_std' : self.output_std  / self.num_graphs,
        }

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