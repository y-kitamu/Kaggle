from typing import Any, List

import tensorflow as tf

import clef
from clef.tasks.base_task import BaseTask


class ClefTask(BaseTask):

    def create_loss_function(self) -> Any:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        return loss_fn

    def create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.optimizer.lr)
        return optimizer

    def create_tfrecords(self) -> None:
        pass

    def create_spectrograms(self, filename) -> List[str]:
        """`filename`の音声ファイルからスペクトログラムを作成する
        """
