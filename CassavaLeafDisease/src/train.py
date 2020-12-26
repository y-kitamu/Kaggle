import os
import re
import math
import argparse
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import callbacks
import tensorflow_addons as tfa
from sklearn.metrics import classification_report

from src.model import get_model
from src.dataset import get_kfold_dataset, get_train_val_dataset
from src.solver import Solver
from src.lr_scheduler import manual_lr_scheduler
from src.callbacks import ProgressLogger
from src.utility import set_gpu
from src.constant import CONFIG_ROOT, OUTPUT_ROOT
from src.predict import evaluate

log = logging.getLogger(__name__)


def get_optimizer(cfg):
    if hasattr(cfg.train, "optimizer"):
        return tf.keras.optimizers.get(dict(**cfg.train.optimizer))
    return Adam()


def get_loss(cfg):
    if hasattr(cfg.train, "loss"):
        if cfg.train.loss.class_name == "focal_loss":
            return tfa.losses.SigmoidFocalCrossEntropy(
                **cfg.train.loss.config, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        else:
            return tf.keras.losses.get(dict(**cfg.train.loss))
    return CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)


def get_lr_scheduler(cfg):
    default_lr = cfg.train.initial_lr
    if cfg.train.lr_schedule.class_name == "manual_lr_scheduler":
        return lambda epoch, idx: manual_lr_scheduler(
            epoch, idx, default_lr=default_lr, **cfg.train.lr_schedule.config)


def restart_at(cfg, model, fold_idx):
    max_epoch = -1
    epoch_regex = re.compile("[0-9][0-9][0-9]+")
    for path in (OUTPUT_ROOT / cfg.title).glob("epoch{}_*.hdf5".format(fold_idx)):
        basename = os.path.basename(path)
        epoch = int(epoch_regex.search(basename).group(0))
        if epoch > max_epoch:
            max_epoch = epoch
    if max_epoch == -1:
        return 0
    weight_path = str(OUTPUT_ROOT / cfg.title / "epoch{}_{:03d}.hdf5".format(fold_idx, max_epoch))
    model.load_weights(weight_path)
    return max_epoch


def log_to_mlflow(cfg, accuracy, metrics):
    # import here because to avoid error when submit in kaggle
    import mlflow

    # define here because to avoid error when submit in kaggle
    def log_params(cfg):
        mlflow.log_params({
            "title": cfg.title,
            "model": cfg.train.model.class_name,
            "optimizer": cfg.train.optimizer.class_name,
            "loss": cfg.train.loss.class_name,
            "epochs": cfg.train.epochs,
            "folds": cfg.train.k_fold,
        })

    def log_metrics(accuracy, metrics):
        mlflow.log_metric("accuracy", accuracy)
        for cls, mtx in metrics.items():
            if isinstance(mtx, float):
                mlflow.log_metric(cls, mtx)
                continue
            for key, val in mtx.items():
                mlflow.log_metric("{}_{}".format(key, cls), val)

    def log_artifacts(cfg):
        for fold_idx in range(cfg.train.k_fold):
            output_dir = OUTPUT_ROOT / cfg.title
            acc_weight = output_dir / "best_val_acc{}.hdf5".format(fold_idx)
            loss_weight = output_dir / "best_val_loss{}.hdf5".format(fold_idx)
            epoch_weight = output_dir / "epoch{}_{:03d}.hdf5".format(fold_idx, cfg.train.epochs)
            mlflow.log_artifact(str(acc_weight), artifact_path=cfg.title)
            mlflow.log_artifact(str(loss_weight), artifact_path=cfg.title)
            mlflow.log_artifact(str(epoch_weight), artifact_path=cfg.title)

    log_params(cfg)
    log_metrics(accuracy, metrics)
    log_artifacts(cfg)


def prepare_callbacks(cfg, fold_idx):
    output_dir = OUTPUT_ROOT / cfg.title
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log.debug("Result output directory : {}".format(output_dir))
    callback_list = []
    # callback_list.append(ProgressLogger())
    callback_list.append(
        callbacks.CSVLogger(os.path.join(output_dir, "./train_log{}.csv".format(fold_idx))))
    callback_list.append(callbacks.LearningRateScheduler(get_lr_scheduler(cfg)))
    callback_list.append(
        callbacks.ModelCheckpoint(os.path.join(output_dir, "best_val_acc{}.hdf5".format(fold_idx)),
                                  save_weights_only=True,
                                  monitor="val_accuracy",
                                  mode="max",
                                  save_best_only=True))
    callback_list.append(
        callbacks.ModelCheckpoint(os.path.join(output_dir, "best_train_acc{}.hdf5".format(fold_idx)),
                                  save_weights_only=True,
                                  monitor="accuracy",
                                  mode="max",
                                  save_best_only=True))
    callback_list.append(
        callbacks.ModelCheckpoint(os.path.join(output_dir, "best_val_loss{}.hdf5".format(fold_idx)),
                                  save_weights_only=True,
                                  monitor="val_loss",
                                  mode="min",
                                  save_best_only=True))
    callback_list.append(
        callbacks.ModelCheckpoint(os.path.join(output_dir, "best_train_loss{}.hdf5".format(fold_idx)),
                                  save_weights_only=True,
                                  monitor="loss",
                                  mode="min",
                                  save_best_only=True))
    callback_list.append(
        callbacks.ModelCheckpoint(os.path.join(output_dir,
                                               "epoch{}_{{epoch:03d}}.hdf5".format(fold_idx)),
                                  save_weights_only=True,
                                  save_best_only=False),)
    return callback_list


def setup(cfg, fold_idx=0):
    set_gpu(cfg.gpu)
    model = get_model(cfg)
    optimizer = get_optimizer(cfg)
    loss = get_loss(cfg)

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    # model.summary()
    callback_list = prepare_callbacks(cfg, fold_idx)
    return model, optimizer, loss, callback_list


def train(cfg):
    title = "============================== Start Train : {} ==============================".format(
        cfg.title)
    log.info("=" * len(title))
    log.info(title)
    log.info("=" * len(title))

    train_batch_size = cfg.train.batch_size
    val_batch_size = train_batch_size * 2

    preds = []
    trues = []
    kf = get_kfold_dataset(cfg)
    for idx, (train_ds, val_ds) in enumerate(kf):
        log.info("==================== Fold : {} / {} ====================".format(
            idx + 1, cfg.train.k_fold))
        model, _, _, callback_list = setup(cfg, idx)
        start_epoch = restart_at(cfg, model, idx)

        log.info("Train data num : {}".format(train_ds.samples))
        log.info("Validation data num : {}".format(val_ds.samples))
        log.info("batch_size : {}".format(cfg.train.batch_size))
        log.info("start epoch / total epoch : {} / {}".format(start_epoch, cfg.train.epochs))

        model.fit(train_ds,
                  epochs=cfg.train.epochs,
                  initial_epoch=start_epoch,
                  steps_per_epoch=math.ceil(train_ds.samples / train_batch_size),
                  validation_data=val_ds,
                  validation_steps=math.ceil(val_ds.samples / val_batch_size),
                  callbacks=callback_list)

        log.info("start evaluation")
        pred, true = evaluate(cfg, val_ds, idx)
        preds.append(pred)
        trues.append(true)

    preds = np.concatenate(preds, axis=0).argmax(axis=1)
    trues = np.concatenate(trues, axis=0).argmax(axis=1)
    log.info("\n{}".format(classification_report(trues, preds)))
    # metrics = classification_report(trues, preds, output_dict=True)
    # accuracy = (preds == trues).sum() / preds.shape[0]
    # log_to_mlflow(cfg, accuracy, metrics)
    log.info("Successfully Finish Training!")


def solve(cfg, train_gen=None, val_gen=None, fold_idx=0):
    if train_gen is None:
        train_gen, val_gen = get_train_val_dataset()
    model, optimizer, loss, callback_list = setup(cfg)
    solver = Solver(cfg=cfg,
                    model=model,
                    train_gen=train_gen,
                    val_gen=val_gen,
                    optimizer=optimizer,
                    loss_func=loss,
                    callbacks=callbacks.CallbackList(callback_list))
    solver.train()


def solve_kfold(cfg):
    kf = get_kfold_dataset(cfg)
    for idx, (train_gen, val_gen) in enumerate(kf):
        solve(cfg, train_gen, val_gen, idx)


def get_config_instance(config_name="config.yaml", config_dir=str(CONFIG_ROOT)):
    # import here because to avoid error when submit in kaggle
    from hydra.experimental import initialize, compose
    relpath = os.path.relpath(config_dir, os.path.dirname(__file__))
    with initialize(config_path=relpath):
        cfg = compose(config_name)
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Cassava Leaf Disease Classification")
    parser.add_argument("--configname", "-c", default="config.yaml")
    parser.add_argument("--configdir", "-d", default=CONFIG_ROOT)
    args = parser.parse_args()

    cfg = get_config_instance(args.configname, args.configdir)
    train(cfg)
