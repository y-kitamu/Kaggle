"""Non local NN module
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import temsorflow_addons as tfa


def sn_conv_reshape(x,
                    out_channel,
                    kernel_size=(1, 1),
                    pool_size=None,
                    pool_strides=2,
                    training=True,
                    name=""):
    """Convolution + Spectral Normalization + (MaxPooling) + reshape.
    Part of self-attention module.
    Args:
        x (tf.Tensor)                   : 4D input tensor ([num batch, height, width, channel])
        out_channel (int)               : Number of output channel
        kernel_size (tuple of int)      : Kernel size of convolution
        pool_size (int or tuple of int) : If None, add no pooling layer,
            Otherwise, add maxpooling layer of pool size = `pool_size` and strides = `pool_strides`
        pool_strides (int)              :
    Return:
        x (tf.tensor) :
    """
    batch_size, height, width, _ = x.shape
    num_location = height * width
    x = tfa.layers.SpectralNormalization(
        layer=Conv2D(out_channel, kernel_size=kernel_size, strides=(1, 1), padding="same"),
        power_iterations=1,
    )(x, training=training)
    if pool_size is not None:
        x = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides)(x)
        num_location /= (pool_strides * pool_strides)
    x = tf.reshape(x, [batch_size, num_location, out_channel])
    return x


def non_local(x, qk_channel_factor=8, v_channel_factor=2, training=True, name=""):
    """Self attention layer.
    Args:
        x (tf.Tensor) : 4D input tensor [Batch, Height, Width, Channel]
        qk_channel_factor (int) : Scale of channel number of Query and Key's Middle layer.
            Number of channels of middle layer is calculated by <num_input_channel> / <channel_factor>
        v_channel_factor (int) : Scale of channel number of Value's Middle layer.
    Return:
        out (tf.Tensor) :
    """
    batch_size, height, width, input_channel = x.shape
    qk_mid_channel = int(input_channel / qk_channel_factor)
    v_mid_channel = int(input_channel / v_channel_factor)

    theta = sn_conv_reshape(x,
                            qk_mid_channel,
                            kernel_size=(1, 1),
                            training=training,
                            name=name + "_query")
    phi = sn_conv_reshape(x,
                          qk_mid_channel,
                          kernel_size=(1, 1),
                          pool_size=(2, 2),
                          training=training,
                          name=name + "_key")

    attn = tf.matmul(theta, phi, transpose_b=True, name=name + "_qk_matmul")
    attn = tf.keras.activations.softmax(attn, name=name + "_qk_sm")
    # print(tf.math.reduce_sum(attn, axis=-1))

    g = sn_conv_reshape(x,
                        v_mid_channel,
                        kernel_size=(1, 1),
                        pool_size=(2, 2),
                        training=training,
                        name=name + "_value")
    out = tf.matmul(attn, g, name=name + "_qkv_matmul")
    out = tf.reshape(out, [batch_size, height, width, v_mid_channel])
    out = tf.keras.layers.Conv2D(input_channel, kernel_size=(1, 1), strides=(1, 1),
                                 padding="same")(out, training=training)

    out = tf.keras.layers.Add()([out, x])
    return out
