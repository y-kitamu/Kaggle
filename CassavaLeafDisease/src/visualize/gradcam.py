import logging

import cv2
import numpy as np
import tensorflow as tf

log = logging.getLogger(__name__)


def gradcam(model, input_image, target_class, layer_name):
    """Create Grad-cam image
    Args:
        model (tf.keras.models.Model) :
        input_image (np.ndarray)      : 3D array of BGR-image.
        target_class (int)            : target class
        layer_name (int or str)       : target layer name or index
    Return:
        np.ndarray : heatmap
        np.ndarray : raw + heatmap image
    """
    layer = None
    try:
        if isinstance(layer_name, str):
            layer = model.get_layer(name=layer_name)
        if isinstance(layer_name, int):
            layer = model.get_layer(index=layer_name)
    except ValueError:
        log.warn("Invalid layer name or index : {}. Skip calcurate gradcam.".format(layer_name))
        return
    grad_model = tf.keras.models.Model(model.input, [layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_output, pred = grad_model(input_image, training=False)
        loss = pred[0, target_class]
    grads = tape.gradient(loss, conv_output)

    kernel_weights = tf.math.reduce_sum(grads[0], axis=[0, 1])
    cam = tf.reshape(kernel_weights, (1, 1, -1)) * conv_output[0]
    cam = tf.math.reduce_sum(cam, axis=2).numpy()
    cam = cv2.resize(cam, input_image.shape[:2])
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    output_image = cv2.addWeighted(cv2.cvtColor((input_image * 255).astype('uint8')), 0.5, cam, 1, 0)
    return heatmap, output_image
