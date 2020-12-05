import os
import numpy as np
import csv
import logging

from src.constant import OUTPUT_ROOT
from src.dataset import TestDatasetGenerator
from src.model import get_model
from src.utility import set_gpu

log = logging.getLogger(__name__)


def get_and_load_model(cfg, model_weights):
    model = get_model(cfg)
    model.load_weights(model_weights)
    return model


def predict(cfg,
            output_filename="./submission.csv",
            model_dir=os.path.join(os.path.dirname(__file__), "../results/baseline/"),
            test_data_dir="../input/cassava-leaf-disease-classification"):
    set_gpu(cfg.gpu)
    test_ds = TestDatasetGenerator(test_data_dir)

    log.info("Loading models...")
    models = [
        get_and_load_model(cfg, model_dir / "{}{}.hdf5".format(metrix, idx))
        for metrix in cfg.test.models
        for idx in range(cfg.train.k_fold)
    ]

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
        csv_writer.writerow(["image_id", "pred"])
        for fname, pred in zip(test_ds.filenames, preds):
            csv_writer.writerow([fname, pred])

    log.info("Successfully finish prediction!")
    return preds
