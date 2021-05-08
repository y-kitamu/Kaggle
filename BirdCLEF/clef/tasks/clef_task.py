from clef.data.tfrecords import write_images_to_tfrecord
from typing import Any, List, TYPE_CHECKING
import random

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

import clef
from clef.tasks.base_task import BaseTask

if TYPE_CHECKING:
    from clef.config import clef_definitions


def convert_audio_to_spectrogram(audio_path, nfft=512, window=1600, stride=1600):
    audio = tfio.audio.AudioIOTensor(str(audio_path))
    audio_tensor = tf.squeeze(audio[:], axis=[-1])
    spectrogram = tfio.experimental.audio.spectrogram(audio_tensor,
                                                      nfft=nfft,
                                                      window=window,
                                                      stride=stride)
    return spectrogram.numpy().T


class ClefTask(BaseTask):

    def __init__(self, config: "clef_definitions.TaskConfig") -> None:
        super().__init__(config)
        self.config = config  # only for type checking
        random.seed(self.config.random_state)

    def create_loss_function(self) -> Any:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        return loss_fn

    def create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.optimizer.lr)
        return optimizer

    def create_tfrecords(self) -> None:
        output_root_dirpath = clef.data.get_tfrecords_dirpath(self.config.train_data)
        output_root_dirpath.mkdir(parents=True, exist_ok=True)
        clef.logger.info("TFRecords output root dir : {}".format(str(output_root_dirpath)))

        short_audio_dirs = [
            path for path in clef.constant.TRAIN_SHORT_AUDIO_PATH.glob("*") if path.is_dir()
        ]

        for dirpath in short_audio_dirs:
            clef.logger.debug("Start process directory : {}".format(str(dirpath)))

            file_list = sorted(list(dirpath.glob("*.ogg")))
            offset = random.randrange(0, len(file_list))
            fold_index_list = [(i + offset) % self.config.num_folds for i in range(len(file_list))]
            random.shuffle(fold_index_list)

            fold_image_list = [[] for _ in range(self.config.num_folds)]
            for fold, file_path in zip(fold_index_list, file_list):
                spec_img = convert_audio_to_spectrogram(file_path, self.config.spectrogram)
                fold_image_list[fold].append(spec_img)

            for fold in range(self.config.num_folds):
                output_train_filepath = output_root_dirpath / "{}_{}_fold{}.tfrecords".format(
                    self.config.train_data.tfrecords_basename, dirpath.name, fold)
                labels = np.ones(len(
                    fold_image_list[fold])) * clef.constant.BIRDNAME_LABEL_DICT[dirpath.name]
                write_images_to_tfrecord(fold_image_list[fold], labels, str(output_train_filepath))
                clef.logger.debug("Save tfrecords file : {}".format(output_train_filepath))
