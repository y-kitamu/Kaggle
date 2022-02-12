"""train.py
"""
import argparse
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
from happy_wheel.dataset.dataloader import SPECIES, build_train_dataloader
from happy_wheel.lr_schedulers import WarmUpCosineAnnealing


def create_model(config: happy_wheel.config.Config) -> keras.Model:
    """ """
    # setup model
    base_model = ycore.models.get_model(config.model.name, **config.model.kwargs)
    inputs = keras.Input(shape=(config.train_dataset.height, config.train_dataset.width, 3))
    x = base_model(inputs)
    x = keras.layers.Conv2D(len(SPECIES), kernel_size=3)(x)
    outputs = keras.layers.GlobalAveragePooling2D(data_format="channels_last")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    # setup optimizer and loss
    optimizer = ycore.optimizers.get_optimizer(config.optimizer.name, **config.optimizer.kwargs)
    loss = ycore.losses.get_loss(config.loss.name, config.loss.kwargs)
    # compile
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            keras.metrics.CategoricalCrossentropy(**config.loss.kwargs),
            keras.metrics.CategoricalAccuracy(),
        ],
    )
    return model


def prepare_callbacks(config: happy_wheel.config.Config) -> List[keras.callbacks.Callback]:
    """ """
    output_dir = RESULT_ROOT / config.exp_name
    output_dir.mkdir(exist_ok=True, parents=True)
    callbacks = [
        keras.callbacks.BackupAndRestore(backup_dir=str(output_dir / "backup")),
        keras.callbacks.CSVLogger(filename=str(output_dir / f"training.csv")),
        keras.callbacks.LearningRateScheduler(WarmUpCosineAnnealing()),
        keras.callbacks.ModelCheckpoint(
            str(output_dir / "best_acc.h5"),
            monitor="val_acc",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            str(output_dir / "best_loss.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.ProgbarLogger(),
        keras.callbacks.TensorBoard(
            str(ycore.TENSORBOARD_LOG_DIR / f"happy_wheel_{config.exp_name}"), profile_batch=10
        ),
    ]
    return callbacks


def run(config: happy_wheel.config.Config):
    """Run training"""
    logger.info("Start training : ")
    logger.info(config.json(indent=2))
    # save settings to file
    config_file = RESULT_ROOT / config.exp_name / "config.json"
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
        steps_per_epoch=(num_data + config.train_dataset.batch_size - 1)
        // config.train_dataset.batch_size,
        # steps_per_epoch=10,
        # validation_steps=10,
    )

    logger.info("Finish training")


def run_cross_validation(
    base_exp_name: str, num_cv_fold: int = 5, image_width: int = 768, image_height: int = 512
):
    """ """
    train_config = TrainConfig(epochs=30)
    model_config = ModelConfig(
        name="efficientnet-b0",
        kwargs=dict(
            include_top=False,
            input_shape=(image_height, image_width, 3),
        ),
    )

    for fold in range(num_cv_fold):
        train_dataset_config = DatasetConfig(
            input_dir=DATA_ROOT / "preprocessed" / "train_images",
            batch_size=8,
            width=image_width,
            height=image_height,
            shuffle=True,
            label_csv_path=DATA_ROOT / "preprocessed" / f"train_cv{fold}.csv",
        )
        val_dataset_config = DatasetConfig(
            input_dir=DATA_ROOT / "preprocessed" / "train_images",
            batch_size=16,
            width=image_width,
            height=image_height,
            shuffle=False,
            label_csv_path=DATA_ROOT / "preprocessed" / f"valid_cv{fold}.csv",
        )
        config = Config(
            exp_name=f"{base_exp_name}/cv{fold}",
            train=train_config,
            model=model_config,
            loss=LossConfig(name="categorical_crossentropy"),
            optimizer=OptimizerConfig(name="adam"),
            train_dataset=train_dataset_config,
            validation_dataset=val_dataset_config,
            test_dataset=val_dataset_config,
        )
        run(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("HappyWheelTrain")
    parser.add_argument("-e", "--exp_name", default="test")
    parser.add_argument("-g", "--gpu", nargs="+", type=int, default=[0])
    args = parser.parse_args()

    ycore.set_gpu(args.gpu)
    run_cross_validation(base_exp_name=args.exp_name)
