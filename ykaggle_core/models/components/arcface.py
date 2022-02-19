"""arcface.py
"""
import typing
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras


class ArcFaceLayer(keras.layers.Layer):
    """ArcFace layer. Use with `ykaggle_core.losses.ArcFaceLoss`.
    Args:
        n_class (int) : Number of output classes.
        margin (float) : Output of GT label class (cos(theta)) is replaced to cos(theta + margin).
        scale (float) : temperature parameter
    """

    def __init__(self, n_class: int, **kwargs):
        super().__init__(**kwargs)
        self.n_class = n_class

    def get_config(self):
        config = super().get_config()
        config.update({"n_class": self.n_class})
        return config

    def build(self, input_shape):
        self.w = self.add_weight(
            name="weights",
            shape=(input_shape[-1], self.n_class),
            initializer="he_normalV2",
        )

    def call(self, inputs: tf.Tensor):
        """Calculate normalized
        Args:
            inputs (tf.Tensor) : 2D input tensor. ([num_batch, num_feature])
        """
        # get norm of inputs
        x = tf.math.l2_normalize(inputs, axis=1)
        # get norm of weights
        w = tf.math.l2_normalize(self.w, axis=0)
        x = tf.matmul(x, w)
        return x
