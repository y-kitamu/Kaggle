"""sample_train.py

Author : Yusuke Kitamura
Create Date : 2022-02-13 14:48:47
Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
"""
import argparse
import shutil
import typing
from typing import List

import ykaggle_core as ycore
from tensorflow import keras

from happy_wheel.config import (Config, DatasetConfig, LossConfig, ModelConfig,
                                OptimizerConfig, TrainConfig)

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras

import happy_wheel
from happy_wheel import DATA_ROOT, RESULT_ROOT, logger
from happy_wheel.train import create_model, prepare_callbacks


def build_train_dataloader(dataset_config: happy_wheel.config.DatasetConfig, is_validation):
    import numpy as np
    import tensorflow as tf
    import tensorflow_datasets as tfds

    (train_ds, val_ds, test_ds), metadata = tfds.load(
        "tf_flowers",
        split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
        with_info=True,
        as_supervised=True,
    )
    target = train_ds if not is_validation else val_ds
    height = dataset_config.height
    width = dataset_config.width
    target = (
        target.map(
            lambda tensor, label: (
                tf.image.resize(tensor, size=[height, width]),
                tf.one_hot(label, 5),
            )
        )
        .batch(64)
        .repeat()
    )

    return target, None


def run(config: happy_wheel.config.Config):
    """Run training"""
    logger.info("Start training : ")
    logger.info(config.json(indent=2))
    # save settings to file
    config_file = RESULT_ROOT / config.exp_name / "config.json"
    if config_file.parent.exists():
        shutil.rmtree(config_file.parent)
    config_file.parent.mkdir(exist_ok=True, parents=True)
    with open(config_file, "w", newline="\n") as f:
        f.write(config.json(indent=2))

    # setup data loader
    train_dataloader, num_data = build_train_dataloader(config.train_dataset, is_validation=False)
    valid_dataloader, _ = build_train_dataloader(config.validation_dataset, is_validation=True)

    # setup train model
    model = create_model(config)

    # prepare callback
    callbacks = prepare_callbacks(config)
    # run train
    model.fit(
        train_dataloader,
        epochs=config.train.epochs,
        validation_data=valid_dataloader,
        callbacks=[callbacks],
        # steps_per_epoch=(num_data + config.train_dataset.batch_size - 1)
        # // config.train_dataset.batch_size,
        steps_per_epoch=200,
        validation_steps=100,
    )

    logger.info("Finish training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("HappyWheelTrain")
    parser.add_argument("-e", "--exp_name", default="test")
    parser.add_argument("-g", "--gpu", nargs="+", type=int, default=[0])
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    args = parser.parse_args()

    ycore.set_gpu(args.gpu)

    image_width = 128
    image_height = 128
    train_config = TrainConfig(epochs=30)
    model_config = ModelConfig(
        name="efficientnet-b0",
        num_output_class=5,
        kwargs=dict(
            include_top=False,
            input_shape=(image_height, image_width, 3),
        ),
    )

    train_dataset_config = DatasetConfig(
        input_dir=DATA_ROOT / "preprocessed" / "train_images",
        batch_size=32,
        width=image_width,
        height=image_height,
        shuffle=True,
    )
    val_dataset_config = DatasetConfig(
        input_dir=DATA_ROOT / "preprocessed" / "train_images",
        batch_size=64,
        width=image_width,
        height=image_height,
        shuffle=False,
    )
    config = Config(
        exp_name=f"sample_test",
        train=train_config,
        model=model_config,
        loss=LossConfig(name="categorical_crossentropy", kwargs=dict(from_logits=True)),
        optimizer=OptimizerConfig(name="adam"),
        train_dataset=train_dataset_config,
        validation_dataset=val_dataset_config,
        test_dataset=val_dataset_config,
    )
    run(config)
