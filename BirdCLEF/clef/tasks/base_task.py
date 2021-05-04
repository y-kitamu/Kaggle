import os
import abc
import glob
from typing import Any, List, Dict, Tuple, Union, TYPE_CHECKING

import numpy as np
import tensorflow as tf

import clef
from clef.constant import PREPROC_DATA_PATH
from clef import config_definitions
from clef.data.tfrecords import create_dataset_from_tfrecord
from clef.callbacks.checkpoint import ModelCheckPoint

if TYPE_CHECKING:
    from clef.callbacks.callback_delegate import CallbackDelegate


class BaseTask():

    loss = "loss"

    def __init__(self, config: config_definitions.TaskConfig, logging_dir: str = None) -> None:
        self.config = config
        self.logging_dir = logging_dir
        self.loss_fn = self.create_loss_function()

    @abc.abstractclassmethod
    def create_loss_function(self) -> Any:
        """Return loss function
        """

    @abc.abstractclassmethod
    def create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Return optimizer
        """

    def create_tfrecords(self) -> None:
        pass

    @abc.abstractclassmethod
    def build_model(self) -> tf.keras.Model:
        """Return model
        """

    def build_inputs(self, is_training: bool) -> tf.data.Dataset:
        config = self.config.train_data if is_training else self.config.validation_data
        tfrecords = clef.data.get_tfrecords_files(config)
        dataset = create_dataset_from_tfrecord(tfrecords)
        if is_training:
            dataset = dataset.shuffle(config.num_data).repeat().batch(self.config.batch_size)
        else:
            dataset = dataset.batch(self.config.batch_size)
        return dataset

    def build_metrics(self, training: bool = True) -> List[tf.metrics.Metric]:
        prefix = "training_" if training else "validation_"
        metrics = [
            tf.metrics.CategoricalAccuracy(prefix + "accuracy"),
            tf.metrics.CategoricalCrossentropy(prefix + "ce_loss")
        ]
        return metrics

    def build_callbacks(self) -> "List[CallbackDelegate]":
        output_dir = "./model"
        callbacks = [
            ModelCheckPoint(metric_name="epoch", output_dir=output_dir),
            ModelCheckPoint(metric_name="validation_accuracy", output_dir=output_dir),
        ]
        return callbacks  # type: ignore

    def process_metrics(self, metrics: List[tf.keras.metrics.Metric],
                        labels: Union[tf.Tensor, np.ndarray], outputs: Union[tf.Tensor, np.ndarray],
                        logs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        if metrics:
            labels = tf.one_hot(labels, self.config.output_classes)
            for metric in metrics:
                metric.update_state(labels, outputs)
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
            loss = self.loss_fn(labels, outputs, from_logits=True)
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
