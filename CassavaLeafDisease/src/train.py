import os
import math
import argparse

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import callbacks
from hydra.experimental import initialize, compose

from src.model import get_model
from src.dataset import get_kfold_dataset, get_train_val_dataset
from src.solver import Solver
from src.lr_scheduler import manual_lr_scheduler
from src.callbacks import ProgressLogger
from src.utility import set_gpu
from src.constant import CONFIG_ROOT, OUTPUT_ROOT


def get_optimizer(cfg):
    if hasattr(cfg["train"], "optimizer"):
        return tf.keras.optimizers.get(dict(**cfg["train"]["optimizer"]))
    return Adam()


def get_loss(cfg):
    if hasattr(cfg["train"], "loss"):
        return tf.keras.losses.get(dict(**cfg["train"]["loss"]))
    return CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)


def get_lr_scheduler(cfg):
    default_lr = cfg["train"]["initial_lr"]
    if cfg["train"]["lr_schedule"]["class_name"] == "manual_lr_scheduler":
        return lambda epoch, idx: manual_lr_scheduler(
            epoch, idx, default_lr=default_lr, **cfg["train"]["lr_schedule"]["config"])


def prepare_callbacks(cfg, fold_idx):
    output_dir = OUTPUT_ROOT / cfg["title"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Result output directory : {}".format(output_dir))
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
        callbacks.ModelCheckpoint(os.path.join(output_dir, "best_val_loss{}.hdf5".format(fold_idx)),
                                  save_weights_only=True,
                                  monitor="val_loss",
                                  mode="min",
                                  save_best_only=True))
    callback_list.append(
        callbacks.ModelCheckpoint(os.path.join(output_dir,
                                               "epoch{}_{{epoch:03d}}.hdf5".format(fold_idx)),
                                  save_weights_only=True,
                                  save_best_only=False),)
    return callback_list


def setup(cfg, fold_idx=0):
    set_gpu(cfg["gpu"])
    model = get_model(cfg)
    optimizer = get_optimizer(cfg)
    loss = get_loss(cfg)

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    model.summary()
    callback_list = prepare_callbacks(cfg, fold_idx)
    return model, optimizer, loss, callback_list


def train(cfg):
    train_batch_size = cfg["train"]["batch_size"]
    val_batch_size = train_batch_size * 2

    kf = get_kfold_dataset(cfg)
    for idx, (train_ds, val_ds) in enumerate(kf):
        print("==================== Fold : {} / {} ====================".format(
            idx + 1, cfg["train"]["k_fold"]))
        model, _, _, callback_list = setup(cfg, idx)
        model.fit(train_ds,
                  epochs=cfg["train"]["epochs"],
                  initial_epoch=cfg["train"]["start_epoch"],
                  steps_per_epoch=math.ceil(train_ds.samples / train_batch_size),
                  validation_data=val_ds,
                  validation_steps=math.ceil(val_ds.samples / val_batch_size),
                  callbacks=callback_list)


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
