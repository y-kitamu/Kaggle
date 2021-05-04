import os
from typing import Any

import tensorflow as tf

import clef
from clef.constant import PREPROC_DATA_PATH
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
        train_record_file = os.path.join(record_dir, "{}_train.tfrecords".format(file_basename))
        write_images_to_tfrecord(x_train, y_train, train_record_file)
        clef.logger.info("Write train data to {}".format(train_record_file))
        test_record_file = os.path.join(record_dir, "{}_test.tfrecordds".format(file_basename))
        write_images_to_tfrecord(x_test, y_test, test_record_file)
        clef.logger.info("Write test data to {}".format(test_record_file))

    def build_model(self) -> tf.keras.Model:
        return create_simple_model(input_shape=self.config.input_shape,
                                   output_classes=self.config.output_classes)
