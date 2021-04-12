import tensorflow as tf

from clef import task
from clef import config_definitions


class Trainer(object):

    def __init__(self, config: config_definitions.TrainerConfig, task: task.MnistTask,
                 steps_per_epoch: int) -> None:
        self._config = config
        self._task = task
        self.steps_per_epoch = steps_per_epoch

        # dataset
        self.train_dataset = self.task.build_inputs(is_training=True)
        self.validation_dataset = self.task.build_inputs(is_training=False)

        # metrics
        self._train_loss = tf.keras.metrics.Mean("training_loss", dtype=tf.float32)
        self._validation_loss = tf.keras.metrics.Mean("validation_loss", dtype=tf.float32)

        self._model = None
        self._optimizer = None

    def compile(self, strategy):
        with strategy:
            self._model = task.build_model()
            self._optimizer = task.create_optimizer()

        self._train_metrics = self._task.build_metrics(training=True) + self._model.metrics
        self._validation_metrics = self._task.build_metrics(training=False) + self._model.metrics

    # @tf.function(jit_compile=True)
    def train(self):
        """train 1-epoch.
        """
        self.on_epoch_begin()
        task_train_step = tf.function(self.task.train_step, jit_compile=True)
        for _ in range(self.steps_per_epoch):
            self.on_step_begin()
            logs = self.strategy.run(task_train_step,
                                     args=(self.train_dataset.take(1), self.model, self.optimizer,
                                           self.train_metrics))
            self.on_step_end(logs)
        self.on_epoch_end()

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_begin(self, logs=None):
        pass

    def on_step_begin(self, logs=None):
        pass

    def on_step_end(self, logs=None):
        self._train_loss.update_state(logs[self.task.loss])
        for metric in self.train_metrics:
            metric.update_state(logs[metric.name])

    def on_epoch_end(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    @property
    def strategy(self):
        return self._strategy

    @property
    def config(self):
        return self._config

    @property
    def task(self):
        return self._task

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer
