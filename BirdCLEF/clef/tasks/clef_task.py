from typing import Any, List, TYPE_CHECKING

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

    def __init__(self, config: clef_definitions.TaskConfig, logging_dir: str) -> None:
        super().__init__(config, logging_dir=logging_dir)
        self.config = config  # only for type checking

    def create_loss_function(self) -> Any:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        return loss_fn

    def create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.optimizer.lr)
        return optimizer

    def create_tfrecords(self) -> None:
        output_root_dirpath = clef.data.get_tfrecords_dirpath(self.config.train_data)
        clef.logger.info("TFRecords output root dir : {}".format(str(output_root_dirpath)))

        short_audio_dirs = [
            path for path in clef.constant.TRAIN_SHORT_AUDIO_PATH.glob("*") if path.is_dir()
        ]

        for dirpath in short_audio_dirs:
            clef.logger.debug("Start process directory : {}".format(str(dirpath)))
            for file_path in dirpath.glob("*.ogg"):
                spec_img = convert_audio_to_spectrogram(file_path, self.config.spectrogram)
