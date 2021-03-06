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
from src.train.solver import Solver
from src.train.lr_scheduler import manual_lr_scheduler
from src.train.callbacks import ProgressLogger, LRScheduler
from src.utility import set_gpu, load_config, run_as_multiprocess, clear_gpu
from src.constant import CONFIG_ROOT, OUTPUT_ROOT
from src.predict import predict, get_and_load_model

log = logging.getLogger(__name__)


def get_optimizer(cfg):
    if hasattr(cfg.train, "optimizer"):
        if cfg.train.optimizer.class_name == "AdamW":
            return tfa.optimizers.AdamW(**cfg.train.optimizer.config)
        else:
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
    log.info("Load latest weights : {}".format(weight_path))
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


def prepare_callbacks(cfg, output_dir, fold_idx):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log.debug("Result output directory : {}".format(output_dir))
    callback_list = []
    # callback_list.append(ProgressLogger())
    callback_list.append(callbacks.ProgbarLogger(count_mode="steps"))
    callback_list.append(
        callbacks.CSVLogger(os.path.join(output_dir, "./train_log{}.csv".format(fold_idx))))
    # callback_list.append(callbacks.LearningRateScheduler(get_lr_scheduler(cfg)))
    callback_list.append(LRScheduler())
    if hasattr(cfg.train, "earlystop_patience"):
        callback_list.append(
            callbacks.EarlyStopping(monitor="val_loss",
                                    patience=cfg.train.earlystop_patience,
                                    restore_best_weights=True))
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


def setup(cfg, output_dir, fold_idx=0):
    model = get_model(cfg)
    # model.summary()
    if hasattr(cfg.train, "transfer_model"):
        log.info("Load weights from {}".format(cfg.train.transfer_model))
        model.load_weights(cfg.train.transfer_model, by_name=True)
    optimizer = get_optimizer(cfg)
    loss = get_loss(cfg)

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    # model.summary()
    callback_list = prepare_callbacks(cfg, output_dir, fold_idx)
    return model, optimizer, loss, callback_list


@run_as_multiprocess
def train_impl(cfg, train_ds, val_ds, output_dir, idx):
    set_gpu(cfg.gpu)
    model, _, _, callback_list = setup(cfg, output_dir, idx)
    start_epoch = restart_at(cfg, model, idx)
    if start_epoch >= cfg.train.epochs:
        log.info("Training is already finished.")
        return
    log.info("Start epoch / Total epoch : {} / {}".format(start_epoch, cfg.train.epochs))

    model.fit(train_ds,
              epochs=cfg.train.epochs,
              initial_epoch=start_epoch,
              steps_per_epoch=math.ceil(len(train_ds) / cfg.train.batch_size),
              validation_data=val_ds,
              validation_steps=math.ceil(len(val_ds) / cfg.train.val_batch_size),
              callbacks=callback_list)


@run_as_multiprocess
def eval_impl(cfg, val_ds, output_dir, idx):
    set_gpu(cfg.gpu)
    models = [
        get_and_load_model(cfg, os.path.join(output_dir, model_name.format(idx)))
        for model_name in ["best_val_acc{}.hdf5", "best_val_loss{}.hdf5"]
    ]
    val_ds.repeat(False)
    pred, true = predict(val_ds, models, cfg.n_classes)
    print("\n{}".format(classification_report(true.argmax(axis=1), pred.argmax(axis=1))))
    clear_gpu(cfg.gpu)


def train(cfg):
    title = "============================== Start Train : {} ==============================".format(
        cfg.title)
    log.info("=" * len(title))
    log.info(title)
    log.info("=" * len(title))

    kf = get_kfold_dataset(cfg)
    output_dir = OUTPUT_ROOT / cfg.title
    for idx, (train_ds, val_ds) in enumerate(kf):
        log.info("==================== Fold : {} / {} ====================".format(
            idx + 1, cfg.train.k_fold))
        log.info("Train data num : {}".format(len(train_ds)))
        log.info("Validation data num : {}".format(len(val_ds)))
        log.info("Image size (width x height) : ({} x {})".format(train_ds.image_width,
                                                                  train_ds.image_height))
        log.info("Batch_size : {}".format(cfg.train.batch_size))

        if cfg.train.model.is_freeze:
            log.info("========== Start transfer learning ==========")
        elif cfg.train.model.is_finetune:
            log.info("========== Start fine tuning ==========")
        train_impl(cfg, train_ds, val_ds, output_dir, idx)

        if cfg.train.model.is_freeze and hasattr(cfg.train.model,
                                                 "is_finetune") and cfg.train.model.is_finetune:
            log.info("========== Start fine tuning ==========")
            cfg.train.model.is_freeze = False
            train_impl(cfg, train_ds, val_ds, output_dir, idx)
            cfg.train.model.is_freeze = True

        log.info("========== Start evaluation ==========")
        eval_impl(cfg, val_ds, output_dir, idx)

    clear_gpu(cfg.gpu)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Cassava Leaf Disease Classification")
    parser.add_argument("--configname", "-c", default="config.yaml")
    parser.add_argument("--configdir", "-d", default=CONFIG_ROOT)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO)
    cfg = load_config(args.configname, args.configdir)
    train(cfg)
