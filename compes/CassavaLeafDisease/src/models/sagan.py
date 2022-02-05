import tensorflow as tf
import tensorflow_addons as tfa

from src.models.non_local import non_local


def sn_conv(x, num_filter, kernel_size, training=True, name=""):
    """Spectral Normalization + Convolution
    Args:
        x (tf.Tensor) :
        num_filter (int) :
        kernel_size (int) :
    Return:
        x (tf.Tensor) :
    """
    x = tfa.layers.SpectralNormalization(
        layer=tf.keras.layers.Conv2D(num_filter,
                                     kernel_size=kernel_size,
                                     padding='same',
                                     name=name + "_conv"),
        power_iteration=1,
        name=name + "_sn",
    )(x, training=training)
    return x


def sn_dense(x, num_filter, training=True, name=""):
    """Spectral Normalization + Dense layer
    Args:
        x (tf.Tensor) :
        num_filter (int) :
    Return:
        x (tf.Tensor) :
    """
    x = tfa.layers.SpectralNormalization(
        layer=tf.keras.layers.Dense(num_filter, name=name + "_dense"),
        power_iteration=1,
        name=name + "_sn",
    )(x, training=training)
    return x


def bn_relu_snconv(x,
                   num_filter,
                   kernel_size=(3, 3),
                   training=True,
                   upsample=False,
                   downsample=False,
                   name=""):
    """Batch normalization + Relu + (spectral normalized) Convolution
    Args:
        x (tf.Tensor) : input tensor (4D-array. shape = [num batch, height, width, num channel])
        num_filter (int) : number of convolution's output filter
    Return:
        x (tf.Tensor) : output tensor (4d-array. shape = [num batch, height, width, num output filter])
    """
    x = tf.keras.layers.BatchNormalization(name=name + "_bn")(x, training=training)
    x = tf.keras.layers.ReLU(name=name + "_relu")(x)
    if upsample:
        x = tf.keras.layers.UpSampling2D(size=(2, 2), name=name + "_upsample")(x)
    x = sn_conv(x, num_filter, kernel_size, training=training, name=name)
    if downsample:
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2),
                                             name=name + "_avg_pool")(x)
    return x


def res_block(inputs, num_filter, training=True, upsample=False, downsample=False, name=""):
    """Residual block.
    Args:
        input (tf.Tensor) :
        num_filter (int)  :
        training (bool)   :
        upsample (bool)   :
        downsample (bool) :
        name (str) :
    Return:
        out (tf.Tensor) :
    """
    x = inputs
    x = bn_relu_snconv(x,
                       num_filter=num_filter,
                       training=training,
                       upsample=upsample,
                       downsample=False,
                       name=name + "0")
    x = bn_relu_snconv(x,
                       num_filter=num_filter,
                       training=training,
                       upsample=False,
                       downsample=downsample,
                       name=name + "1")

    skip = inputs
    if upsample:
        skip = tf.keras.layers.UpSampling2D(size=(2, 2), name=name + "_skip_upsample")(skip)
    if downsample:
        skip = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
                                                strides=(2, 2),
                                                name=name + "_skip_avg_pool")(skip)
    skip = sn_conv(skip, num_filter, kernel_size=(1, 1), training=training, name=name + "_skip")
    out = tf.keras.layers.Add(name=name + "_add")([skip, x])
    return out


def generator(zs, gf_dim, target_class, num_classes, training=True):
    """
    Args:
       zs (tf.Tensor) : input random tensor (2d-array. shape = [num batch, channel])
       target_class (int) :
       num_classes (int)  :
       training (bool)    :
    Return:
       out (tf.Tensor) :
    """
    act0 = sn_dense(zs, num_filter=gf_dim * 16 * 4 * 4, training=training, name="g_sn_dense0")
    act0 = tf.reshape(act0, [-1, 4, 4, gf_dim * 16], training, name="g_block0")
    act1 = res_block(act0, gf_dim * 16, training, upsample=True, name="g_block1")
    act2 = res_block(act1, gf_dim * 8, training, upsample=True, name="g_block2")
    act3 = res_block(act2, gf_dim * 4, training, upsample=True, name="g_block3")
    act4 = res_block(act3, gf_dim * 2, training, upsample=True, name="g_block4")
    act4 = non_local(act4, training=training, name="g_non_local4")
    act5 = res_block(act4, gf_dim, training, upsample=True, name="g_block5")
    act5 = tf.keras.layers.BatchNormalization(name="g_block5_bn")(act5, training=training)
    act5 = tf.keras.layers.ReLU(name="g_block5_relu")(act5)
    act6 = sn_conv(act5, num_filter=3, kernel_size=(3, 3), training=training, name="g_sn_conv6")
    out = tf.nn.tanh(act6, name="g_out_tanh")
    return out


def discriminator(images, labels, df_dim, training=True):
    """
    Args:
        images (tf.Tensor) : input image tensor (4d-array. shape = [num batch, height, width, channel])
        labels (tf.Tensor) : The corresponding labels for the images. (2d-array. one-hot)
        df_dim (int) : factor of discriminator filter dimension.
        training (bool) :
    Return:
        out (tf.Tensor) :
    """
    h0 = res_block(images, df_dim, downsample=True, training=training, name="d_block0")
    h1 = res_block(h0, df_dim * 2, downsample=True, training=training, name="d_block1")
    h2 = res_block(h1, df_dim * 4, downsample=True, training=training, name="d_block2")
    h2 = non_local(h2, training=training, name="d_non_local2")
    h3 = res_block(h2, df_dim * 8, downsample=True, training=training, name="d_block3")
    h4 = res_block(h3, df_dim * 16, downsample=True, training=training, name="d_block4")
    h5 = res_block(h4, df_dim * 16, downsample=False, training=training, name="d_blcok5")
    h5_act = tf.nn.relu(h5, name="d_act5")
    h6 = tf.reduce_sum(h5_act, axis=[1, 2])
    out = sn_dense(h6, 1, training=training, name="d_out")
    h_labels = sn_dense(labels, df_dim * 16, training=training, name="d_label")  # TODO : use embegging?
    out += tf.reduce_sum(h6 * h_labels, axis=1, keepdims=True)
    return out
