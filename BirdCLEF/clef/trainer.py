from typing import Optional, List

import tensorflow as tf

from clef import task
from clef import config_definitions
from clef.callbacks.callback_delegate import CallbackDelegate
from clef.callbacks.logger import Logger
from clef.callbacks.progressbar import ProgressBar
from clef.callbacks.callback_list import CallbackList


class Trainer(object):

    epoch = "epoch"
    step = "step"

    default_callbacks = [ProgressBar(), Logger()]

    def __init__(self,
                 config: config_definitions.TrainerConfig,
                 task: task.MnistTask,
                 callbacks: List[CallbackDelegate] = []) -> None:
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

        # callbacks
        self.callback_list = CallbackList(self.default_callbacks + callbacks)
        self.callback_list.set_trainer(self)

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
        self.callback_list.on_train_begin(logs)

    def on_epoch_begin(self, logs=None) -> None:
        self.callback_list.on_epoch_begin(logs)

        # reset metrics
        self.train_loss.reset_states()
        for metric in self.train_metrics.values():
            metric.reset_states()

    def on_step_begin(self, logs=None) -> None:
        self.callback_list.on_step_begin(logs)

    def on_step_end(self, logs=None) -> None:
        self.callback_list.on_step_end(logs)

    def on_epoch_end(self, logs=None) -> None:
        self.callback_list.on_epoch_end(logs)

    def on_train_end(self, logs=None) -> None:
        self.callback_list.on_train_end(logs)

    def on_validation_begin(self, logs=None) -> None:
        self.callback_list.on_validation_begin(logs)

        # reset metrics
        self.validation_loss.reset_states()
        for metric in self.validation_metrics.values():
            metric.reset_states()

    def on_validation_end(self, logs=None) -> None:
        self.callback_list.on_validation_end(logs)

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
