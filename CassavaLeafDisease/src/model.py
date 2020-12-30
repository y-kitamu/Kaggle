import logging

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import \
    Convolution2D, GlobalAveragePooling2D, BatchNormalization, ReLU, Dropout, Dense
from tensorflow.keras.regularizers import l2

log = logging.getLogger(__name__)


def conv2d_bn_activation(x, output_filters, kernel_size=(2, 2), strides=(2, 2), weight_decay=0.0):
    x = Convolution2D(filters=output_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      kernel_regularizer=l2(weight_decay),
                      padding="same",
                      use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def get_base_model(cfg, inputs):
    model = None
    if cfg.train.model.class_name == "efficientnetb0":
        model = tf.keras.applications.EfficientNetB0(include_top=False,
                                                     weights=None,
                                                     input_tensor=inputs,
                                                     pooling=None)
    if cfg.train.model.class_name == "efficientnetb4":
        model = tf.keras.applications.EfficientNetB4(include_top=False,
                                                     weights=None,
                                                     input_tensor=inputs,
                                                     pooling=None)
    if hasattr(cfg.train.model, "is_freeze") and cfg.train.model.is_freeze:
        log.info("Freeze base model : {}".format(cfg.train.model.class_name))
        model.trainable = False
    if hasattr(cfg.train.model, "is_finetune") and cfg.train.model.is_finetune:
        log.info("Fine tune model : {}".format(cfg.train.model.class_name))
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
    return model


def get_head_output(cfg, inputs, weight_decay=0.0, dropout_rate=0.0, **kwargs):
    x = conv2d_bn_activation(inputs,
                             output_filters=320,
                             kernel_size=(1, 1),
                             strides=(1, 1),
                             weight_decay=weight_decay)
    if dropout_rate > 1e-5:
        x = Dropout(rate=dropout_rate)(x)
    x = Convolution2D(filters=cfg.n_classes,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      kernel_regularizer=l2(weight_decay),
                      padding="same",
                      use_bias=True)(x)
    outputs = GlobalAveragePooling2D()(x)
    return outputs


def get_model(cfg):
    input_shape = (cfg.image_height, cfg.image_width, cfg.n_channel)
    inputs = Input(shape=input_shape)
    base_model = get_base_model(cfg, inputs)

    kwargs = {} if not hasattr(cfg.train.model, "config") else cfg.train.model.config
    outputs = get_head_output(cfg, base_model.output, **kwargs)
    model = Model(inputs, outputs, name=cfg.train.model.class_name)
    return model
