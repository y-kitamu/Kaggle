import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import \
    Convolution2D, GlobalAveragePooling2D, BatchNormalization, ReLU, Dropout, Dense
from tensorflow.keras.regularizers import l2


def conv2d_bn_activation(x, output_filters, kernel_size=(2, 2), strides=(2, 2), weight_decay=0.0):
    x = Convolution2D(filters=output_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      kernel_regularizer=l2(weight_decay),
                      padding="valid",
                      use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def efficientnetb0(cfg, weight_decay=0.0, dropout_rate=0.2):
    input_shape = (cfg.image_height, cfg.image_width, cfg.n_channel)
    inputs = Input(shape=input_shape)
    efn = tf.keras.applications.EfficientNetB0(include_top=False,
                                               weights=None,
                                               input_tensor=inputs,
                                               pooling=None)

    x = conv2d_bn_activation(efn.output,
                             output_filters=640,
                             kernel_size=(2, 2),
                             strides=(2, 2),
                             weight_decay=weight_decay)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(rate=dropout_rate)(x)
    outputs = Dense(cfg.n_classes, kernel_regularizer=l2(weight_decay))(x)
    model = Model(inputs, outputs, name="EfficientNetB0")
    return model


def get_model(cfg):
    if cfg.train.model.class_name == "efficientnetb0":
        kwargs = {} if not hasattr(cfg.train.model, "config") else cfg.train.model.config
        return efficientnetb0(cfg, **kwargs)
