"""lr_schedulers.py
"""


import math


class WarmUpCosineAnnealing:
    """Learning rate scheduler (Warm up + Cosine Annealing).
    Args:
        warmup_epoch (int) :
        wamup_lr (float) :
        cycle_epoch (int) :
        max_lr (float) :
        min_lr (float) :
    """

    def __init__(
        self,
        warmup_epoch: int = 1,
        warmup_lr: float = 1e-5,
        cycle_epoch: int = 10,
        max_lr: float = 5e-3,
        min_lr: float = 1e-6,
    ):
        self.warmup_epoch = warmup_epoch
        self.wamup_lr = warmup_lr
        self.cycle_epoch = cycle_epoch
        self.max_lr = max_lr
        self.min_lr = min_lr

    def __call__(self, epoch, lr) -> float:
        """Calculate learning rate at `epoch`."""
        if epoch <= self.warmup_epoch:
            return self.wamup_lr
        radian = (epoch - self.warmup_epoch) / self.cycle_epoch
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(radian))
        return lr
