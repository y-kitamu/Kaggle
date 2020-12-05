import math

import numpy as np

from src.constant import OUTPUT_ROOT
from src.model import get_model
from src.utility import set_gpu


def get_and_load_model(cfg, model_weights):
    model = get_model(cfg)
    model.load_weights(model_weights)
    return model


def evaluate(cfg, dataset, fold_idx=0, weights=["best_val_acc", "best_val_loss"]):
    set_gpu(cfg.gpu)

    model_dir = OUTPUT_ROOT / cfg.title
    models = [
        get_and_load_model(cfg, model_dir / model_dir / "{}{}.hdf5".format(weight, fold_idx))
        for weight in weights
    ]

    preds = np.zeros((dataset.samples, cfg.n_classes))
    labels = np.zeros((dataset.samples, cfg.n_classes))
    max_idx = math.ceil(dataset.samples / dataset.batch_size)
    start = 0
    for idx, (imgs, ls) in enumerate(dataset):
        # print("\r {} / {}".format(idx * dataset.batch_size, dataset.samples), end="")
        pred = sum([model(imgs, training=False) for model in models])
        end = start + pred.shape[0]
        preds[start:end] = pred.numpy()[:]
        labels[start:end] = ls[:]
        start = end
        if idx == max_idx:
            break
    preds /= len(models)
    return preds, labels
