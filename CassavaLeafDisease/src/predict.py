import os
import csv
import logging

import numpy as np

from src.model import get_model
from src.utility import set_gpu

log = logging.getLogger(__name__)


def get_and_load_model(cfg, model_weights):
    if not os.path.exists(model_weights):
        log.warning("Model weights does not exist. Skip loading : {}".format(model_weights))
        return None
    model = get_model(cfg)
    model.load_weights(model_weights)
    return model


def predict_for_submission(cfg,
                           dataset,
                           output_filename="./submission.csv",
                           model_dir=os.path.join(os.path.dirname(__file__), "../results/baseline/"),
                           model_weights=["best_val_acc0.hdf5"]):
    """Prediction script for submission
    Args:
        cfg (OmegaConf.DefaultDict)                : Configurations imported from yaml file by hydra
        dataset (src.dataset.TestDatasetGenerator) :
        output_filename (str)                      : Output csv filename. If None, no output is saved.
        model_dir (str)                            : Model direcotry.
        model_weights (list of str)                : Model weights to be used.
            This function calculate predictons per each model and take the average of them.
        test_data_dir (str)                        : target files
    """
    set_gpu(cfg.gpu)
    log.info("Loading models...")
    models = [get_and_load_model(cfg, os.path.join(model_dir, basename)) for basename in model_weights]
    models = [model for model in models if model is not None]

    scores, _ = predict(dataset, models, cfg.n_classes)
    preds = scores.argmax(axis=1)

    if output_filename is not None:
        log.info("Finish prediction. Write result to csv.")
        with open(output_filename, 'w') as f:
            csv_writer = csv.writer(f, lineterminator="\n")
            csv_writer.writerow(["image_id", "label"])
            for fname, pred in zip(dataset.filenames, preds):
                csv_writer.writerow([fname, pred])

    log.info("Successfully finish prediction!")
    return scores, dataset.labels, dataset.filenames


def predict(dataset, models, n_classes=5):
    """Predict dataset using models.
    Args:
        dataset (iterable)                                :
        models (list of tf.keras.Model or tf.keras.Model) :
        n_classes (int)                                   :
    Retrun:
        np.ndarray : 2D array of predictions ([Num Samples, Num Classes])
    """
    log.info("Start prediction")
    log.info("Evaluate data num : {}".format(dataset.samples))
    log.info("Batch size        : {}".format(dataset.batch_size))
    if not isinstance(models, (list, tuple)):
        models = [models]

    dataset.with_label = False
    dataset.repeat(False)
    preds = np.zeros((len(dataset), n_classes), dtype=np.float32)
    for model_idx, model in enumerate(models):
        start = 0
        for idx, imgs in enumerate(dataset):
            print("\r {} / {}    ".format(idx * dataset.batch_size, dataset.samples), end="")
            end = start + imgs.shape[0]
            preds[start:end] += model(imgs, training=False)
            start = end
        print("\nFinish {} / {}".format(model_idx + 1, len(models)))
    preds /= len(models)
    return preds, dataset.labels
