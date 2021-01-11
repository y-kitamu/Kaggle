import os
import csv
import logging
import argparse

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
    log.info("Evaluate data num : {}".format(len(dataset)))
    log.info("Batch size        : {}".format(dataset.batch_size))
    if not isinstance(models, (list, tuple)):
        models = [models]

    dataset.with_label = False
    dataset.repeat(False)
    preds = np.zeros((len(dataset), n_classes), dtype=np.float32)
    for model_idx, model in enumerate(models):
        start = 0
        for idx, imgs in enumerate(dataset):
            print("\r {} / {}    ".format(idx * dataset.batch_size, len(dataset)), end="")
            end = start + imgs.shape[0]
            preds[start:end] += model(imgs, training=False)
            start = end
        print("\nFinish {} / {}".format(model_idx + 1, len(models)))
    preds /= len(models)
    return preds, dataset.labels


if __name__ == "__main__":
    import glob

    from sklearn.metrics import classification_report

    from src.constant import TRAIN_DATA_DIR, TRAIN_CSV, CONFIG_ROOT, OUTPUT_ROOT
    from src.utility import load_config
    from src.dataset import get_train_val_dataset

    parser = argparse.ArgumentParser("Cassava Leaf Disease Prediction")
    parser.add_argument("-c", "--configname", default="config.yaml")
    parser.add_argument("-d", "--configdir", default=CONFIG_ROOT)
    parser.add_argument("-i", "--inputdir", default=TRAIN_DATA_DIR)
    parser.add_argument("--inputcsv", default=TRAIN_CSV)
    parser.add_argument("--models", nargs="+", default=["best_val_acc*.hdf5", "best_val_loss*.hdf5"])
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO)
    cfg = load_config(args.configname, args.configdir)
    dataset, _ = get_train_val_dataset(cfg,
                                       data_dir=args.inputdir,
                                       csv_fname=args.inputcsv,
                                       is_train=False)
    model_dir = os.path.join(OUTPUT_ROOT, cfg.title)
    model_weights = [
        os.path.basename(fname)
        for model_name in args.models
        for fname in glob.glob(os.path.join(model_dir, model_name))
    ]
    scores, labels, _ = predict_for_submission(cfg,
                                               dataset,
                                               output_filename=None,
                                               model_dir=model_dir,
                                               model_weights=model_weights)
    if len(scores.shape) == 2:
        scores = scores.argmax(axis=1)
    if len(labels.shape) == 2:
        labels = labels.argmax(axis=1)
    classification_report(labels, scores)
