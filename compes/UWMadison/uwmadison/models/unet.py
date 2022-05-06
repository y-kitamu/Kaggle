"""unet.py

Author : Yusuke Kitamura
Create Date : 2022-05-06 22:18:11
Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
"""
from typing import TYPE_CHECKING

import tensorflow as tf
from tensorflow import keras

if TYPE_CHECKING:
    from keras.api._v2 import keras


class ContractingBlock(keras.layers.Layer):
    """Conv -> Relu -> Conv -> Relu -> downsample"""

    def __init__(self, n_channel: int):
        self.conv1 = keras.layers.Conv2D(
            n_channel, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv2 = keras.layers.Conv2D(
            n_channel, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.pool = keras.layers.MaxPool2D()

    def call(self, inputs: tf.Tensor):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool(x)
        return x


class ExpandingBlock(keras.layers.Layer):
    """Conv -> Relu -> Conv -> Relu -> upsample"""

    def __init__(self, n_channel):
        self.conv1 = keras.layers.Conv2D(
            n_channel, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.conv2 = keras.layers.Conv2D(
            n_channel, kernel_size=(3, 3), padding="same", activation="relu"
        )
        self.pool = keras.layers.Conv2DTranspose(
            n_channel / 2, kernel_size=3, strides=2, padding="same"
        )

    def call(self, inputs: tf.Tensor):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool(x)
        return x


class UNet(keras.Model):
    def __init__(self, n_class):
        self.cblock1 = ContractingBlock(64)
        self.cblock2 = ContractingBlock(128)
        self.cblock3 = ContractingBlock(256)
        self.cblock4 = ContractingBlock(512)
        self.eblock4 = ExpandingBlock(1024)
        self.eblock3 = ExpandingBlock(512)
        self.eblock2 = ExpandingBlock(256)
        self.eblock1 = ExpandingBlock(128)
        self.conv1 = keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")
        self.conv2 = keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")
        self.conv3 = keras.layers.Conv2D(n_class, kernel_size=1, padding="same")

    def call(self, inputs: tf.Tensor):
        x1 = self.cblock1(inputs)
        x2 = self.cblock2(x1)
        x3 = self.cblock3(x2)
        x4 = self.cblock4(x3)
        x = self.eblock4(x4)
        x = keras.layers.concatenate([x, x4], axis=-1)
        x = self.eblock3(x)
        x = keras.layers.concatenate([x, x3], axis=-1)
        x = self.eblock2(x)
        x = keras.layers.concatenate([x, x2], axis=-1)
        x = self.eblock1(x)
        x = keras.layers.concatenate([x, x1], axis=-1)
        return x
