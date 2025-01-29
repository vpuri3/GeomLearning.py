#
import os
import torch
import shutil
from tqdm import tqdm

import mlutils

from .time_march import march_case
from .visualize import visualize_timeseries_pyv

__all__ = [
    'FinaltimeCallback',
    'TimeseriesCallback',
]

#======================================================================#
class Callback:
    def __init__(self, case_dir: str, mesh: bool, save_every=None):
        self.case_dir = case_dir
        self.mesh = mesh
        self.save_every = save_every

    def get_ckpt_dir(self, trainer: mlutils.Trainer, final: bool):
        if final:
            ckpt_dir = os.path.join(self.case_dir, f'final')
        else:
            ckpt_dirs = [dir for dir in os.listdir(self.case_dir) if dir.startswith('ckpt')]
            nsave = len(ckpt_dirs) #+ 1
            ckpt_dir = os.path.join(self.case_dir, f'ckpt{str(nsave).zfill(2)}')

        if os.path.exists(ckpt_dir):
            print(f"Removing {ckpt_dir}")
            shutil.rmtree(ckpt_dir)

        return ckpt_dir

    def load(self, trainer: mlutils.Trainer):
        ckpt_dirs = [dir for dir in os.listdir(self.case_dir) if dir.startswith('ckpt')]
        if len(ckpt_dirs) == 0:
            if trainer.LOCAL_RANK == 0:
                print(f'No checkpoint found in {self.case_dir}. starting from scrach.')
            return
        load_dir = sorted(ckpt_dirs)[-1]
        model_file = os.path.join(self.case_dir, load_dir, 'model.pt')

        trainer.load(model_file)

        return

    def modify_dataset_transform(self, trainer: mlutils.Trainer, val: bool):
        """
        modify transform to return mesh, original fields
        """
        for data in [trainer._data, trainer.data_]:
            if data is None:
                continue
            if isinstance(data, torch.utils.data.Subset):
                data = data.dataset

            data.transform.mesh = True if val else self.mesh
            data.transform.orig = val
            data.transform.elems = val
            data.transform.metadata = val

        return

    def __call__(self, trainer: mlutils.Trainer, final: bool=False):

        #------------------------#
        if trainer.LOCAL_RANK != 0:
            return
        #------------------------#
        if not final:
            if self.save_every is None:
                self.save_every = trainer.stats_every
            if (trainer.epoch % self.save_every) != 0:
                return
        #------------------------#

        ckpt_dir = self.get_ckpt_dir(trainer, final)
        print(f"saving checkpoint to {ckpt_dir}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # save model
        trainer.save(os.path.join(ckpt_dir, 'model.pt'))

        # update data transform
        self.modify_dataset_transform(trainer, True)

        # evaluate model
        self.evaluate(trainer, ckpt_dir)

        # revert data transform
        self.modify_dataset_transform(trainer, False)

        return

#======================================================================#
class FinaltimeCallback(Callback):
    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str):

        device = trainer.device
        model  = trainer.model.module if trainer.DDP else trainer.model

        for (dataset, split) in zip([trainer._data, trainer.data_], ['train', 'test']):
            if dataset is None:
                print(f"No {split} dataset.")
            else:
                print(f"Evaluating {split} dataset.")

            r2file = os.path.join(ckpt_dir, f'{split}_r2.txt')
            with open(r2file, 'w') as f:
                f.write('Case\tName\tMSE\tR-Square\n')
                for icase in tqdm(range(len(dataset))):
                    data = dataset[icase].to(device)
                    yh = model(data)

                    case_name = data.metadata['case_name']
                    l2 = torch.nn.MSELoss()(yh, data.y).item()
                    r2 = mlutils.r2(yh, data.y)

                    f.write(f'{icase}\t{case_name}\t{l2}\t{r2}\n')

                    del data, yh

        return

#======================================================================#
class TimeseriesCallback(Callback):
    def __init__(
        self, case_dir: str, save_every=None, 
        autoreg_start=1, num_eval_cases=None
    ):
        super().__init__(case_dir, save_every)
        self.autoreg_start = autoreg_start
        self.num_eval_cases = num_eval_cases

    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str):

        device = trainer.device
        model  = trainer.model.module if trainer.DDP else trainer.model

        _data = trainer._data
        data_ = trainer.data_

        _transform = _data.transform
        transform_ = data_.transform if data_ is not None else None

        K = self.autoreg_start
        _C = min(self.num_eval_cases, len(_data.case_files))
        C_ = min(self.num_eval_cases, len(data_.case_files)) if data_ is not None else None

        _cases = [_data[_data.case_range(c)] for c in range(_C)]
        cases_ = [data_[data_.case_range(c)] for c in range(C_)] if data_ is not None else None

        for (cases, transform, split) in zip(
            [_cases, cases_], [_transform, transform_], ['train', 'test'],
        ):
            if cases is None:
                print(f"No {split} dataset.")
            else:
                print(f"Evaluating {split} dataset.")

            for icase in tqdm(range(len(cases))):
                case = cases[icase]
                case_data = cases[icase]
                ii = str(icase).zfill(2)
    
                for (autoreg, ext) in zip([True, False], ['AR', 'NR']):
                    eval_data, l2s, r2s = march_case(
                        model, case_data, transform,
                        autoreg=autoreg, device=device, K=K,
                    )
    
                    name = f'{os.path.basename(self.case_dir)}-{split}{ii}-{ext}'
                    case_dir = os.path.join(ckpt_dir, name)
                    visualize_timeseries_pyv(eval_data, case_dir, merge=True, name=name)
    
                    r2_file = os.path.join(ckpt_dir, f'{name}.txt')
                    with open(r2_file, 'w') as f:
                        f.write('Step\tMSE\tR-Square\n')
                        for (k, (l2,r2)) in enumerate(zip(l2s, r2s)):
                            f.write(f'{k}\t{l2s[k]}\t{r2s[k]}\n')

        return

#======================================================================#
