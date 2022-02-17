"""arcface.py
"""
import typing
from typing import Optional

import tensorflow as tf
from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras


class ArcFace(keras.layers.Layer):
    """ArcFace layer
    Args:
        n_class (int) : Number of output classes.
        margin (float) : Output of GT label class (cos(theta)) is replaced to cos(theta + margin).
    """

    def __init__(self, n_class: int, margin: float, **kwargs):
        super().__init__(**kwargs)
        self.n_class = n_class
        self.margin = tf.constant(margin)
        self._constant = tf.constant(1e-9)

    def build(self, input_shape):
        self.w = self.add_weight(
            name="weights", shape=(input_shape[-1], self.n_class), initializer="he_normalV2"
        )

    def call(self, inputs: tf.Tensor, y_true: Optional[tf.Tensor] = None):
        """Calculate arcface.
        Args:
            inputs (tf.Tensor) :
            y_true (Optional[tf.Tensor]) : one-hot vector representing ground-truth label.
        """
        # get norm of inputs
        x = tf.math.square(inputs)
        x = inputs / (tf.math.sqrt(tf.math.reduce_sum(x, axis=-1, keepdims=True)) + self._constant)
        # get norm of weights
        w = tf.math.square(self.w)
        w = self.w / (tf.math.sqrt(tf.math.reduce_sum(w, axis=0, keepdims=True)) + self._constant)
        x = tf.matmul(x, w)

        if y_true is not None:
            y_margin = y_true * self.margin
            x = tf.acos(x)
            x = tf.math.cos(x + y_margin)

        return x
