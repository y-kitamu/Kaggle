"""Learning rate schedulers
"""


def manual_lr_scheduler(epoch, idx, default_lr, warmup_epoch, annealing_epoch, annealing_scale,
                        num_annealing_step):
    if epoch < warmup_epoch:
        return default_lr * 0.01

    epoch_per_step = int(annealing_epoch / num_annealing_step)
    current_step = int((epoch - warmup_epoch) % annealing_epoch / epoch_per_step) - 1
    lr = (annealing_scale**current_step) * default_lr
    print("new lr : {}".format(lr))
    return lr
