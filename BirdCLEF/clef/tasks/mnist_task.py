from typing import Any

import tensorflow as tf

from clef.tasks import BaseTask
from clef.model import create_simple_model


class MnistTask(BaseTask):

    def create_loss_function(self) -> Any:
        return tf.keras.losses.sparse_categorical_crossentropy

    def create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        optimizer = tf.keras.optimizers.Adam()
        return optimizer

    def build_model(self) -> tf.keras.Model:
        return create_simple_model(input_shape=self.config.input_shape,
                                   output_classes=self.config.output_classes)
