"""arcface_loss.py

Author : Yusuke Kitamura
Create Date : 2022-02-19 13:25:02
Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
"""
import math
import typing
from typing import Tuple

import tensorflow as tf
from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras


class ArcFaceLoss(keras.losses.Loss):
    """ArcFace loss. Use with `ykaggle_core.models.components.ArcFaceLayer`.
    Args:
        margin (float) : margin added to gt class prediction. (Unit : radian)
        scale (float) : temperature scaling
    """

    def __init__(self, margin: float = 0.5, scale: float = 64.0, **kwargs):
        super().__init__(**kwargs)
        self.margin = tf.constant(margin, dtype=tf.float32)
        self.scale = tf.constant(scale, dtype=tf.float32)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """
        Args:
            y_true (tf.Tensor) : one-hot vector representing ground-truth label (individual).
                2d-tensor ([Batch, N_class])
            y_pred (tf.Tensor) : 2d-tensor prediction. ([Batch, N_class])
        """
        # assert y_true.shape == y_pred.shape
        y_margin = y_true * self.margin
        x = tf.acos(y_pred)
        x = tf.clip_by_value(x + y_margin, clip_value_min=0, clip_value_max=math.pi)
        x = tf.math.cos(x)
        x = x * self.scale

        loss = tf.math.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y_true, x))
        return loss
