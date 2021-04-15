from typing import Optional
import time

import tensorflow as tf

import clef
from clef import task
from clef import config_definitions


class Trainer(object):

    epoch = "epoch"
    step = "step"

    def __init__(self, config: config_definitions.TrainerConfig, task: task.MnistTask) -> None:
        self._config = config
        self._task = task
        self.steps_per_epoch = self._task.config.steps_per_epoch

        # dataset
        self.train_dataset = self.task.build_inputs(is_training=True)
        self.validation_dataset = self.task.build_inputs(is_training=False)

        # metrics
        self.train_loss = tf.keras.metrics.Mean("training_loss", dtype=tf.float32)
        self.validation_loss = tf.keras.metrics.Mean("validation_loss", dtype=tf.float32)

        self._model = None
        self._optimizer = None

    def compile(self, strategy: tf.distribute.Strategy) -> None:
        with strategy.scope():
            self._model = self.task.build_model()
            self._optimizer = self.task.create_optimizer()

        self.train_metrics = self.task.build_metrics(training=True)
        self.validation_metrics = self.task.build_metrics(training=False)

        self.train_dataset = strategy.experimental_distribute_dataset(self.train_dataset)
        self.validation_dataset = strategy.experimental_distribute_dataset(self.validation_dataset)

    def train(self, strategy: tf.distribute.Strategy) -> None:
        """train 1-epoch.
        """
        task_train_step = tf.function(self.task.train_step)  #, jit_compile=True)
        train_itr = iter(self.train_dataset)
        for step, inputs in zip(range(self.steps_per_epoch), train_itr):
            self.on_step_begin({self.step: step})
            logs = strategy.run(task_train_step,
                                args=(inputs, self.model, self.optimizer, self.train_metrics))
            self.train_loss.update_state(strategy.reduce("sum", logs[self.task.loss], axis=0))
            self.on_step_end()

    def validation(self, strategy: tf.distribute.Strategy) -> None:
        """validation
        """
        for inputs in self.validation_dataset:
            logs = strategy.run(self.task.validation_step,
                                args=(inputs, self.model, self.validation_metrics))  # yapf: disable
            self.validation_loss.update_state(strategy.reduce("sum", logs[self.task.loss], axis=0))

    def on_train_begin(self, logs=None) -> None:
        clef.logger.info("Start Training.")

    def on_epoch_begin(self, logs=None) -> None:
        self.epoch_begin = time.time()
        clef.logger.info("".join(["=" for _ in range(40)]))
        clef.logger.info("Epoch {}".format(logs[self.epoch]))

        # reset metrics
        self.train_loss.reset_states()
        for metric in self.train_metrics.values():
            metric.reset_states()

    def on_step_begin(self, logs=None) -> None:
        stats = ", ".join([
            "{} = {:.3f}".format(key,
                                 value.result().numpy()) for key, value in self.train_metrics.items()
        ])
        print("\r {} / {} ({})".format(logs[self.step], self.steps_per_epoch, stats), end="")

    def on_step_end(self, logs=None) -> None:
        pass

    def on_epoch_end(self, logs=None) -> None:
        print("\r", end="")
        self.epoch_end = time.time()

        header = "|".join(["{:7s}".format(""), "{:10s}".format("Loss")] +
                          ["{:10s}".format(key) for key in self.train_metrics.keys()])
        train = "|".join([
            "{:7s}".format("Train"), "{:10s}".format("{:.3f}".format(self.train_loss.result().numpy()))
        ] + [
            "{:10s}".format("{:.3f}".format(val.result().numpy()))
            for val in self.train_metrics.values()
        ])
        valid = "|".join([
            "{:7s}".format("Valid"), "{:10s}".format("{:.3f}".format(
                self.validation_loss.result().numpy()))
        ] + [
            "{:10s}".format("{:.3f}".format(val.result().numpy()))
            for val in self.validation_metrics.values()
        ])
        clef.logger.info(header)
        clef.logger.info(train)
        clef.logger.info(valid)
        clef.logger.info("Elapsed Time : {:.2f} second".format(self.epoch_end - self.epoch_begin))

    def on_train_end(self, logs=None) -> None:
        clef.logger.info("End Training")

    def on_validation_begin(self, logs=None):
        self.validation_loss.reset_states()
        for metric in self.validation_metrics.values():
            metric.reset_states()

    def on_validation_end(self, logs=None):
        pass

    @property
    def config(self) -> config_definitions.TrainerConfig:
        return self._config

    @property
    def task(self) -> task.MnistTask:
        return self._task

    @property
    def model(self) -> Optional[tf.keras.Model]:
        return self._model

    @property
    def optimizer(self) -> Optional[tf.keras.optimizers.Optimizer]:
        return self._optimizer
