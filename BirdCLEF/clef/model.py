from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, GlobalAveragePooling2D, Dense, Dropout


def create_simple_model(input_shape: Tuple[int, int, int],
                        output_classes: int,
                        activation: tf.keras.layers.Activation = tf.keras.activations.relu,
                        dropout_rate: float = 0.2) -> tf.keras.Model:
    """
    Args:
        input_shape (tuple) : input image shape (height, width, depth)
        output_classes (int) : number of output classes
        activation (tf.keras.layers.Activation) : activation function
    Return:
        model (tf.keras.Model) :
    """
    input = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation=activation, use_bias=True)(input)
    x = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation=activation, use_bias=True)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(rate=dropout_rate)(x)
    output = Dense(output_classes)(x)
    model = tf.keras.Model(inputs=input, outputs=output)
    return model
