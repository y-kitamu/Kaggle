import os
import glob
from typing import Optional, Any, List, Dict, Tuple, Union

import numpy as np
import tensorflow as tf

from clef import config_definitions
from clef.data import create_dataset_from_tfrecord
from clef.model import create_simple_model


class MnistTask():

    loss = "loss"

    def __init__(self,
                 config: config_definitions.TaskConfig,
                 logging_dir: str = None,
                 name: str = None) -> None:
        self.config = config
        self.logging_dir = logging_dir
        self.name = name
        self.loss_fn = self.create_loss_function()

    def create_loss_function(self) -> Any:
        return tf.keras.losses.sparse_categorical_crossentropy

    def create_optimizer(self) -> Any:
        optimizer = tf.keras.optimizers.Adam()
        return optimizer

    def create_tfrecords(self) -> None:
        pass

    def build_model(self) -> tf.keras.Model:
        return create_simple_model(self.config.input_shape, self.config.output_classes)

    def build_inputs(self,
                     is_training: bool,
                     input_context: Optional[tf.distribute.InputContext] = None) -> tf.data.Dataset:
        config = self.config.train_data if is_training else self.config.validation_data
        tfrecords = glob.glob(
            os.path.join(config.tfrecords_dir, "{}*.tfrecords".format(config.tfrecords_basename)))
        dataset = create_dataset_from_tfrecord(tfrecords)
        if is_training:
            dataset = dataset.shuffle(1000).repeat().batch(self.config.batch_size)
        else:
            dataset = dataset.batch(self.config.batch_size)
        return dataset

    def build_metrics(self, training: bool = True) -> List[tf.metrics.Metric]:
        metrics = [tf.metrics.Accuracy(), tf.metrics.CategoricalCrossentropy()]
        return metrics

    def process_metrics(self, metrics, labels, outputs, logs):
        if metrics:
            for metric in metrics:
                metric.update_state(tf.one_hot(labels, self.config.output_classes), outputs)
            logs.update({m.name: m.result() for m in metrics})
        return logs

    def train_step(self, inputs: Tuple[Any, Any], model: tf.keras.Model,
                   optimizer: tf.keras.optimizers.Optimizer,
                   metrics: List[tf.keras.metrics.Metric]) -> Dict[str, tf.Tensor]:
        """train 1 step.
        Args:
            inputs (tuple) :
            model (tf.keras.Model) :
            optimizer (tf.keras.optimizers.Optimizer) :
            metrics (list) :
        Return:
            logs (dict) :
        """
        images, labels = inputs

        with tf.GradientTape() as tape:
            outputs = model(images, training=True)
            loss = self.loss_fn(labels, outputs)
            scaled_loss = loss / tf.distribute.get_strategy().num_replicas_in_sync
        grads = tape.gradient(scaled_loss, model.trainable_variables)

        optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))

        logs = {self.loss: loss}
        logs = self.process_metrics(metrics, labels, outputs, logs)
        return logs

    def validation_step(self, inputs: Tuple[Any, Any], model: tf.keras.Model,
                        metrics: List[tf.keras.metrics.Metric]) -> Dict[str, tf.Tensor]:
        """validation 1 step
        Args:
            inputs (tuple) : tuple of (input images, labels).
                images = 4d-array ([B, H, W, C]), label = 2d-array ([B, categorical label])
            model (tf.keras.Model) :
            metrics (list) :
        Return:
            logs (dict) :
        """
        images, labels = inputs
        outputs = self.inference_step(images, model)
        loss = self.loss_fn(labels, outputs)

        logs = {self.loss: loss}
        logs = self.process_metrics(metrics, labels, outputs, logs)
        return logs

    def inference_step(self, inputs: Union[tf.Tensor, np.ndarray],
                       model: tf.keras.Model) -> Union[tf.Tensor, np.ndarray]:
        """inference inputs
        Args:
            inputs (tf.Tensor or np.ndarray) : input images (4d-array [B, H, W, C])
            model (tf.keras.Tensor) :
        Return:
            (tf.Tensor or np.ndarray) : model output
        """
        return model(inputs, training=False)
