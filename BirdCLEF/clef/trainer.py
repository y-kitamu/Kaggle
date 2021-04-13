from typing import Optional
import time

import tensorflow as tf

import clef
from clef import task
from clef import config_definitions


class Trainer(object):

    epoch = "epoch"

    def __init__(self, config: config_definitions.TrainerConfig, task: task.MnistTask,
                 steps_per_epoch: int) -> None:
        self._config = config
        self._task = task
        self.steps_per_epoch = steps_per_epoch

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

        self.train_metrics = self.task.build_metrics(training=True) + self._model.metrics
        self.validation_metrics = self.task.build_metrics(training=False) + self._model.metrics

    # @tf.function(jit_compile=True)
    def train(self, epoch: int, strategy: tf.distribute.Strategy) -> None:
        """train 1-epoch.
        """
        self.on_epoch_begin({self.epoch: epoch})
        # task_train_step = tf.function(self.task.train_step)  #, jit_compile=True)
        task_train_step = self.task.train_step
        train_itr = iter(self.train_dataset)
        for _, inputs in zip(range(self.steps_per_epoch), train_itr):
            self.on_step_begin()
            logs = strategy.run(task_train_step,
                                args=(inputs, self.model, self.optimizer, self.train_metrics))
            self.on_step_end(logs)

    def validation(self, epoch: int, strategy: tf.distribute.Strategy) -> None:
        for metric in self.validation_metrics:
            metric.reset_states()

        for inputs in self.validation_dataset:
            logs = strategy.run(self.task.validation_step,
                                args=(inputs, self.model, self.validation_metrics))  # yapf: disable

    def on_train_begin(self, logs=None) -> None:
        clef.logger.info("Start Training.")

    def on_epoch_begin(self, logs=None) -> None:
        self.epoch_begin = time.time()
        clef.logger.info("Epoch {}".format(logs[self.epoch]))

    def on_step_begin(self, logs=None) -> None:
        pass

    def on_step_end(self, logs=None) -> None:
        self.train_loss.update_state(logs[self.task.loss])

    def on_epoch_end(self, logs=None) -> None:
        self.epoch_end = time.time()
        clef.logger.info("Elapsed Time : {} second".format(self.epoch_end - self.epoch_begin))

    def on_train_end(self, logs=None) -> None:
        clef.logger.info("End Training")

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
