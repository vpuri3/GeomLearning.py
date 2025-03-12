#
import os
import json

import gc
import torch
import shutil
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import mlutils

__all__ = [
    'Callback',
    'TSCallback',
]

#======================================================================#
class Callback:
    def __init__(self, case_dir: str, save_every=None):
        self.case_dir = case_dir
        self.save_every = save_every
        self.final = False

    def get_ckpt_dir(self, trainer: mlutils.Trainer):
        if self.final:
            ckpt_dir = os.path.join(self.case_dir, f'final')
        else:
            nsave = trainer.epoch // self.save_every
            ckpt_dir = os.path.join(self.case_dir, f'ckpt{str(nsave).zfill(2)}')

        if os.path.exists(ckpt_dir) and trainer.GLOBAL_RANK == 0:
            print(f"Removing {ckpt_dir}")
            shutil.rmtree(ckpt_dir)

        return ckpt_dir

    def load(self, trainer: mlutils.Trainer):
        ckpt_dirs = [dir for dir in os.listdir(self.case_dir) if dir.startswith('ckpt')]
        if len(ckpt_dirs) == 0:
            if trainer.GLOBAL_RANK == 0:
                print(f'No checkpoint found in {self.case_dir}. starting from scrach.')
            return
        load_dir = sorted(ckpt_dirs)[-1]
        model_file = os.path.join(self.case_dir, load_dir, 'model.pt')

        trainer.load(model_file)

        return

    @torch.no_grad()
    def __call__(self, trainer: mlutils.Trainer, final: bool=False):

        #------------------------#
        self.final = final
        if not self.final:
            if self.save_every is None:
                self.save_every = trainer.stats_every
            if trainer.epoch == 0:
                # return
                pass
            if (trainer.epoch % self.save_every) != 0:
                return
        #------------------------#

        # save model
        ckpt_dir = self.get_ckpt_dir(trainer)
        if trainer.GLOBAL_RANK == 0:
            print(f"saving checkpoint to {ckpt_dir}")
            os.makedirs(ckpt_dir, exist_ok=True)
            trainer.save(os.path.join(ckpt_dir, 'model.pt'))

        # save stats
        if trainer.GLOBAL_RANK == 0:
            print(f"Saving stats to {ckpt_dir}/stats.json")
            with open(os.path.join(ckpt_dir, 'stats.json'), 'w') as f:
                json.dump(trainer.stat_vals, f)

        # evaluate model
        self.evaluate(trainer, ckpt_dir)

        # revert self.final
        self.final = False

        return
    
    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str):
        pass

#======================================================================#
class TSCallback(Callback):
    # def __init__(self, case_dir: str, save_every=None):
    #     super().__init__(case_dir, save_every)

    @torch.no_grad()
    def evaluate(self, trainer: mlutils.Trainer, ckpt_dir: str):
        
        trainer.model.eval()
        
        N, MSE = 0, 0.0
        slice_weights = [[] for _ in range(trainer.model.num_layers)]
        temperature = [[] for _ in range(trainer.model.num_layers)]
        bias = [[] for _ in range(trainer.model.num_layers)]
        attn_weights = [[] for _ in range(trainer.model.num_layers)]

        for batch in trainer._loader_:
            x = batch[0].to(trainer.device)
            y = batch[1].to(trainer.device)
            yh, swt, tmp, bs, att = trainer.model(x, return_stats=True)
            
            n = trainer.get_batch_size(batch)
            N += n
            MSE += ((yh - y).pow(2).mean() * n).item()
            for i in range(trainer.model.num_layers):
                slice_weights[i].append(swt[i].detach().cpu())
                temperature[i].append(tmp[i].detach().cpu())
                bias[i].append(bs[i].detach().cpu())
                attn_weights[i].append(att[i].detach().cpu())
                
            del x, y, yh, swt, tmp, att

        MSE = MSE / N
        slice_weights = [torch.cat(slice_weights[i], dim=0) for i in range(trainer.model.num_layers)]
        attn_weights = [torch.cat(attn_weights[i], dim=0) for i in range(trainer.model.num_layers)]
        temperature = [temperature[i][0] for _ in range(trainer.model.num_layers)]
        bias = [bias[i][0] for _ in range(trainer.model.num_layers)]

        # Attention utilization (number of non-zero attn weights)
        attn_utilization = [(w > 1e-2).sum(dim=-1).float().mean().item() / trainer.model.num_slices for w in attn_weights ]                
        
        print()
        print(f"Train MSE: {MSE:.8f}")
        print(f"Noise level: {trainer.noise_schedule.get_current_noise()}")
        print(f"Attn utilization: {[round(s, 4) for s in attn_utilization]}. Mean: {sum(attn_utilization) / len(attn_utilization):.4f}")
        print()

        # print(f"Temperature stats:")
        # print(f"  Mean: {[round(t.mean().item(), 4) for t in temperature]}")
        # print(f"  Std : {[round(t.std().item(), 4) for t in temperature]}")
        # print(f"  Min : {[round(t.min().item(), 4) for t in temperature]}")
        # print(f"  Max : {[round(t.max().item(), 4) for t in temperature]}")

        print(f"Bias stats:")
        print(f"  Mean: {[round(b.mean().item(), 4) for b in bias]}")
        print(f"  Std : {[round(b.std().item(), 4) for b in bias]}")
        print(f"  Min : {[round(b.min().item(), 4) for b in bias]}")
        print(f"  Max : {[round(b.max().item(), 4) for b in bias]}")

        # Slice sparsity: what proportion of slices are invoked per point
        slice_sparsity = [(w < 1e-2).sum(dim=-2).float().mean().item() * 100 / trainer.model.num_slices for w in slice_weights]
        
        # Slice utilization (how evenly are slices used)
        target_use = 1 / trainer.model.num_slices
        threshold = 0.5 * target_use
        slice_use = [w.mean(dim=-1) for w in slice_weights] # [B H M]
        underused = [(slice_use[i] < (target_use - threshold)).float().mean().item() * 100 for i in range(trainer.model.num_layers)]
        overused = [(slice_use[i] > (target_use + threshold)).float().mean().item() * 100 for i in range(trainer.model.num_layers)]

        # print(f"Slice utilization: How evenly are slices used?")
        # print(f"Want every slice to be used equally often, avoiding scenarios where")
        # print(f"some slices are always ignored (underutilized) or overly relied upon (overutilized).")
        # print(f"Sparsity: What proportion of slices are invoked per point?")

        print()
        print(f"Slice utilization stats:")
        print()
        print(f"  Mean: {[round(s.mean().item(), 4) for s in slice_use]} (Target: {target_use:.5f})")
        print(f"  Std : {[round(s.std(dim=-1).mean().item(), 4) for s in slice_use]}")
        print(f"  Min : {[round(s.min().item(), 4) for s in slice_use]}")
        print(f"  Max : {[round(s.max().item(), 4) for s in slice_use]}")
        print(f"  % Underused (< {target_use - threshold:.5f}): {[round(u, 2) for u in underused]}. Mean: {sum(underused) / len(underused):.4f}")
        print(f"  % Overused  (> {target_use + threshold:.5f}): {[round(o, 2) for o in overused]}. Mean: {sum(overused) / len(overused):.4f}")
        print(f"  % Sparsity : {[round(s, 2) for s in slice_sparsity]}. Mean: {sum(slice_sparsity) / len(slice_sparsity):.4f}")
        print()

        mean_slice_use = [w.mean(dim=[0,-1]) / target_use for w in slice_weights]
        slice_use_bias_corr = [torch.corrcoef(torch.stack([mean_slice_use[i].view(-1), bias[i].view(-1)])) for i in range(trainer.model.num_layers) ]

        print(f"Slice use/ bias correlation: {[round(c[0,1].item(), 4) for c in slice_use_bias_corr]}")
        print()

        fig, axes = plt.subplots(trainer.model.num_layers, 2, figsize=(10, 3*trainer.model.num_layers))
        fig.suptitle('Slice Utilization, Bias')
        
        for i in range(trainer.model.num_layers):

            corr = slice_use_bias_corr[i][0,1].item()

            ax = axes[i, 0]
            im = ax.imshow(mean_slice_use[i].cpu().numpy(), cmap='viridis', aspect='auto', vmin=0, vmax=3)
            ax.set_title(f'Layer {i}: underused: {underused[i]:.1f}% / overused: {overused[i]:.1f}%')
            ax.set_xlabel('Slice Index')
            ax.set_ylabel('Head Index')
            fig.colorbar(im, ax=ax)
            im.cmap.set_over('red')
            im.cmap.set_under('blue')
            
            ax = axes[i, 1]
            im = ax.imshow(bias[i].cpu().numpy(), cmap='viridis', aspect='auto', vmin=-5, vmax=5)
            ax.set_title(f'Layer {i}: bias (min: {bias[i].min().item():.1f}, max: {bias[i].max().item():.1f}, corr: {corr:.3f})')
            ax.set_xlabel('Slice Index')
            ax.set_ylabel('Head Index')
            fig.colorbar(im, ax=ax)
            im.cmap.set_over('red')
            im.cmap.set_under('blue')

        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, 'slice_utilization_and_bias.png'))
        plt.savefig(os.path.join(ckpt_dir, '..', 'slice_utilization_and_bias.png'))

        # OBSERVATIONS:
        # - Negative Biases = Overused Keys
        #   - Confirmed: Negative correlation between bias and usage.
        #   - Why Dead Keys?:
        #     - Initial Skew: Pre-bias dots heavily favor a few keys (e.g., due to poor initialization or point cloud structure). Bias overcorrects these, but underused keys don’t get enough boost.
        #   - Fix:
        #     - Increase noise_scale early (e.g., 0.5 → decay) to disrupt initial skew.
        #     - Use smaller bias step (e.g., 0.0005) to avoid over-penalizing.
        # - Dead Keys Have Neutral/Slight Negative Bias
        #   - Confirmed: Dead keys (usage ≈ 0) don’t have large negative biases.
        #   - Why Dead?:
        #     - Attention Collapse: Top-k locks onto a subset early, and noise/bias can’t recover others.
        #   - Fix:
        #     - Add dropout to attn (e.g., F.dropout(attn, p=0.1) post-softmax) to force exploration.
        #     - Reinitialize q_proj, k_proj with smaller variance (e.g., Xavier init).
        # - Bias Isn’t Balancing
        #   - Confirmed: Low correlation, biases drift without affecting usage.
        #   - Fix:
        #     - Normalize bias periodically (self.bias -= self.bias.mean(dim=-1, keepdim=True)).
        #     - Clamp tighter (e.g., -2 to 2) to keep bias impactful but controlled.
        # 
        
        # if trainer.is_cuda:
        #     gc.collect()
        #     torch.cuda.empty_cache()

        return

#======================================================================#
#