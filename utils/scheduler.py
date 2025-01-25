import numpy as np
from torch.optim.lr_scheduler import LambdaLR


class CosineAnnealingScheduler(LambdaLR):
    def __init__(self, optimizer, total_steps, lr_base, lr_min, lr_max=1):
        """
        Cosine Annealing Scheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            total_steps (int): Total number of training steps.
            lr_min (float): Minimum learning rate at the end of annealing.
            lr_max (float): Maximum learning rate at the start of annealing.
        """
        self.total_steps = total_steps
        self.lr_base = lr_base
        self.lr_min = lr_min / self.lr_base
        self.lr_max = lr_max

        # Define the lambda function for learning rate adjustment
        lr_lambda = lambda step: self.cosine_annealing(step)
        super().__init__(optimizer, lr_lambda=lr_lambda)

    def cosine_annealing(self, step):
        """Cosine Annealing for learning rate decay scheduler"""
        return self.lr_min + (self.lr_max - self.lr_min) * 0.5 * (
                1 + np.cos(step / self.total_steps * np.pi))
