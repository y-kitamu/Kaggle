import numpy as np


def mixup(x, t, lam):
    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    t_a, t_b = t, t[index]
    return mixed_x, t_a, t_b
