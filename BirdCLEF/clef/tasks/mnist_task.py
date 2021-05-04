import os
from typing import Any

import tensorflow as tf

import clef
from clef.data.tfrecords import write_images_to_tfrecord
from clef.tasks import BaseTask
from clef.model import create_simple_model


class MnistTask(BaseTask):

    def create_loss_function(self) -> Any:
        return tf.keras.losses.sparse_categorical_crossentropy

    def create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        optimizer = tf.keras.optimizers.Adam()
        return optimizer

    def create_tfrecords(self) -> None:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # save train data
        record_dir = clef.data.get_tfrecords_dirpath(self.config.train_data)
        train_record_file = os.path.join(
            record_dir, "{}.tfrecords".format(self.config.train_data.tfrecords_basename))
        write_images_to_tfrecord(x_train, y_train, train_record_file)
        clef.logger.info("Write train data to {}".format(train_record_file))

        # save validation data
        record_dir = clef.data.get_tfrecords_dirpath(self.config.validation_data)
        test_record_file = os.path.join(
            record_dir, "{}.tfrecordds".format(self.config.validation_data.tfrecords_basename))
        write_images_to_tfrecord(x_test, y_test, test_record_file)
        clef.logger.info("Write test data to {}".format(test_record_file))

    def build_model(self) -> tf.keras.Model:
        return create_simple_model(input_shape=self.config.input_shape,
                                   output_classes=self.config.output_classes)
