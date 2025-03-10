#
import math
import torch

__all__ = [
    'NoiseScheduler',
]

#======================================================================#
class NoiseScheduler:
    def __init__(
        self,
        initial_noise=0.5,
        min_noise=0.0,
        total_steps=10000,
        decay_type='cosine',
    ):
        assert decay_type in ['linear', 'cosine', 'exp', 'step']
        assert 0 <= initial_noise <= 1
        assert 0 <= min_noise <= 1
        assert 0 <= total_steps

        self.initial_noise = initial_noise  # Starting noise level
        self.min_noise = min_noise          # Minimum noise level
        self.total_steps = total_steps      # Total training steps
        self.decay_type = decay_type        # 'linear', 'cosine', 'exp', 'step'
        self.step_num = 0                   # Current step
        
    def reset(self):
        self.step_num = 0
        
    def set_current_step(self, step_num):
        self.step_num = step_num
        
    def step(self):
        self.step_num += 1

    def get_current_noise(self):
        progress = self.step_num / self.total_steps
        
        if self.decay_type == 'linear':
            noise = self.initial_noise * (1 - progress) + self.min_noise * progress
        elif self.decay_type == 'cosine':
            noise = self.min_noise + 0.5 * (self.initial_noise - self.min_noise) * (1 + math.cos(math.pi * progress))
        elif self.decay_type == 'exp':
            min_noise = math.fabs(self.min_noise) + 1e-6
            decay_rate = -math.log(min_noise / self.initial_noise) / self.total_steps
            noise = self.initial_noise * math.exp(-decay_rate * self.step_num)
        elif self.decay_type == 'step':
            # Example: halve every 25% of steps
            steps_per_drop = self.total_steps // 4
            drop_factor = 0.5 ** (self.step_num // steps_per_drop)
            noise = self.initial_noise * drop_factor
        else:
            raise ValueError(f"Unknown decay_type: {self.decay_type}")
        
        return max(noise, self.min_noise)  # Ensure noise doesn’t go below min

#======================================================================#
if __name__ == '__main__':
    scheduler = NoiseScheduler(
        initial_noise=0.5, min_noise=0.0,
        total_steps=10000, decay_type='cosine',
    )

    for step in range(0, 10001):
        scheduler.step()
        if step % 1000 == 0:
            noise = scheduler.get_current_noise()
            print(f"Step {step}: Noise = {noise:.4f}")

#======================================================================#
#

