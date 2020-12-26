import os
import csv
import logging

import numpy as np
import tensorflow as tf

from src.constant import OUTPUT_ROOT
from src.dataset import TestDatasetGenerator
from src.model import get_model
from src.utility import set_gpu

log = logging.getLogger(__name__)


def get_and_load_model(cfg, model_weights):
    if not os.path.exists(model_weights):
        log.warn("Model weights does not exist. Skip loading : {}".format(model_weights))
        return None
    model = get_model(cfg)
    model.load_weights(model_weights)
    return model


def predict(cfg,
            output_filename="./submission.csv",
            model_dir=os.path.join(os.path.dirname(__file__), "../results/baseline/"),
            model_weights=["best_val_acc0.hdf5"],
            test_data_dir="../input/cassava-leaf-disease-classification"):
    """Prediction script for submission
    Args:
        cfg (OmegaConf.DefaultDict) : Configurations imported from yaml file by using  hydra
        output_filename (str)       : Output csv filename. If None
        model_dir (str)             : Model direcotry.
        model_weights (list of str) : Model weights to be used.
            This function calculate predictons per each model and take the average of them.
        test_data_dir (str)         : target files
    """
    set_gpu(cfg.gpu)
    test_ds = TestDatasetGenerator(test_data_dir)
    log.info("Loading models...")
    models = [get_and_load_model(cfg, os.path.join(model_dir, basename)) for basename in model_weights]
    models = [model for model in models if model is not None]

    log.info("Start prediction")
    preds = np.zeros((len(test_ds), cfg.n_classes))
    start = 0
    for idx, imgs in enumerate(test_ds):
        print("\r {}".format(idx), end="")
        end = start + imgs.shape[0]
        preds[start:end] = sum([model(imgs, training=False) for model in models])
        start = end

    log.info("Finish prediction. Write result to csv.")
    preds = preds.argmax(axis=1)
    with open(output_filename, 'w') as f:
        csv_writer = csv.writer(f, lineterminator="\n")
        csv_writer.writerow(["image_id", "label"])
        for fname, pred in zip(test_ds.filenames, preds):
            csv_writer.writerow([fname, pred])

    log.info("Successfully finish prediction!")
    return preds, test_ds.filenames
