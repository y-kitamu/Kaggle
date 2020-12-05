import math

import numpy as np

from src.constant import OUTPUT_ROOT
from src.model import get_model
from src.utility import set_gpu


def evaluate(cfg, dataset, fold_idx=0):
    set_gpu(cfg.gpu)

    acc_model = get_model(cfg)
    loss_model = get_model(cfg)
    epoch_model = get_model(cfg)

    acc_weight = OUTPUT_ROOT / cfg.title / "best_val_acc{}.hdf5".format(fold_idx)
    loss_weight = OUTPUT_ROOT / cfg.title / "best_val_loss{}.hdf5".format(fold_idx)
    epoch_weight = OUTPUT_ROOT / cfg.title / "epoch{}_{:03d}.hdf5".format(fold_idx, cfg.train.epochs)
    acc_model.load_weights(acc_weight)
    loss_model.load_weights(loss_weight)
    epoch_model.load_weights(epoch_weight)

    preds = np.zeros((dataset.samples, cfg.n_classes))
    labels = np.zeros((dataset.samples, cfg.n_classes))
    max_idx = math.ceil(dataset.samples / dataset.batch_size)
    start = 0
    for idx, (imgs, ls) in enumerate(dataset):
        # print("\r {} / {}".format(idx * dataset.batch_size, dataset.samples), end="")
        pred = acc_model(imgs, training=False)
        pred += loss_model(imgs, training=False)
        pred += epoch_model(imgs, training=False)
        pred /= 3.0
        end = start + pred.shape[0]
        preds[start:end] = pred.numpy()[:]
        labels[start:end] = ls[:]
        start = end
        if idx == max_idx:
            break

    return preds, labels
