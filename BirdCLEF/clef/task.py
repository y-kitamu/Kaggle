import os
import glob
from typing import Optional, Any, List, Dict, Tuple

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
        return tf.keras.losses.categorical_crossentropy

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
        dataset = dataset.shuffle(1000).repeat().batch(self.config.batch_size)
        return dataset

    def build_metrics(self, training: bool = True) -> List[tf.metrics.Metric]:
        metrics = [tf.metrics.Accuracy, tf.metrics.CategoricalCrossentropy]
        return metrics

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
        grads = tape.gradients(scaled_loss, model.trainable_variables)
        optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))

        logs = {self.loss: loss}
        if metrics:
            for metric in metrics:
                metric.update_state(labels, outputs)
            logs.update({m.name: m.result() for m in metrics})
        return logs
