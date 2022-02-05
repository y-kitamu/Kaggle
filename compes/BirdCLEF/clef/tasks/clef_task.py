from typing import Any, Callable, List, Optional, TYPE_CHECKING
import csv
import random
from collections import namedtuple
import multiprocessing as mp
import queue

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

import clef
from clef.tasks.base_task import BaseTask

if TYPE_CHECKING:
    from clef.config import clef_definitions

DATA_INFO = namedtuple("DATA_INFO", ["image_file", "src_file", "label_idx", "label_name", "fold"])


def convert_audio_to_spectrogram(audio_path, nfft=512, window=1600, stride=1600):
    audio = tfio.audio.AudioIOTensor(str(audio_path))
    audio_tensor = tf.squeeze(audio[:], axis=[-1])
    spectrogram = tfio.experimental.audio.spectrogram(audio_tensor,
                                                      nfft=nfft,
                                                      window=window,
                                                      stride=stride)
    return spectrogram.numpy().T


def process_directory(dirpath, output_root_dirpath, config):
    output_file_list = []
    sound_file_list = sorted(list(dirpath.glob("*.ogg")))
    offset = random.randrange(0, len(sound_file_list))
    fold_index_list = [(i + offset) % config.num_folds for i in range(len(sound_file_list))]
    random.shuffle(fold_index_list)
    for fold, file_path in zip(fold_index_list, sound_file_list):
        output_dirpath = output_root_dirpath / dirpath.name
        output_dirpath.mkdir(parents=True, exist_ok=True)
        spec_img = convert_audio_to_spectrogram(file_path,
                                                nfft=config.spectrogram.num_fft,
                                                window=config.spectrogram.fft_window,
                                                stride=config.spectrogram.fft_stride)
        train_images = self.split_spectrogram(spec_img)

        for idx, image in enumerate(train_images):
            output_path = output_dirpath / "{}_{:03d}.png".format(file_path.stem, idx)
            cv2.imwrite(str(output_path), image)
            output_file_list.append(
                DATA_INFO(str(output_path), str(file_path),
                          clef.constant.BIRDNAME_LABEL_DICT[dirpath.name], dirpath.name, fold))
        clef.logger.debug("Finish process file = {}".format(file_path))
    return output_file_list


def data_process_func(data: DATA_INFO, n_classes: int, dtype: tf.dtypes.DType):
    """
    """
    img = cv2.imread(data)
    input_image = tf.constant(img, dtype=dtype, name="Input_Image")
    label = tf.one_hot(data.label_idx, n_classes)
    yield input_image, label


def dataset_generator(data_list: List[DATA_INFO],
                      target_fold: List[int],
                      n_classes: int = len(clef.constant.BIRDNAME_LABEL_DICT),
                      preprocess_func: Optional[Callable] = data_process_func,
                      dtype: tf.dtypes.DType = tf.float32,
                      num_process: int = 8,
                      num_prefetch: int = 256):
    """
    """
    data_list = [data for data in data_list if data.fold in target_fold]
    pool = mp.Pool(num_process)
    que = queue.Queue(maxsize=num_prefetch)
    try:
        while True:
            random.shuffle(data_list)
            for data in data_list:
                if que.full():
                    yield que.get().get()
                que.put(pool.apply_async(preprocess_func, (data, n_classes, dtype)))
    finally:
        pool.terminate()


class ClefTask(BaseTask):

    def __init__(self, config: "clef_definitions.TaskConfig") -> None:
        super().__init__(config)
        self.config = config  # only for type checking
        self.fold_counter = 0
        random.seed(self.config.random_state)

    def create_loss_function(self) -> Any:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        return loss_fn

    def create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.optimizer.lr)
        return optimizer

    def build_inputs(self, is_training: bool) -> tf.data.Dataset:

        if is_training:
            dataset = dataset.shuffle(self.config.train_data.num_data).repeat().batch(
                self.config.batch_size)
        else:
            dataset = dataset.batch(self.config.batch_size)
        return dataset

    def preprocess(self) -> None:
        """
        """
        output_root_dirpath = clef.data.get_data_dirpath(self.config.train_data)
        output_root_dirpath.mkdir(parents=True, exist_ok=True)
        clef.logger.info("TFRecords output root dir : {}".format(str(output_root_dirpath)))

        short_audio_dirs = sorted(
            [path for path in clef.constant.TRAIN_SHORT_AUDIO_PATH.glob("*") if path.is_dir()])

        pool = mp.Pool(8)
        que = queue.Queue()
        for dirpath in short_audio_dirs:
            clef.logger.debug("Start process directory : {}".format(str(dirpath)))
            que.put(pool.apply_async(process_directory, (dirpath, output_root_dirpath, self.config)))

        output_file_list: List[DATA_INFO] = []
        while not que.empty():
            output_file_list += que.get().get()

        output_csv_path = output_root_dirpath / "data.csv"
        with open(output_csv_path, "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["image_path", "source_path", "label_idx", "label_name", "fold"])
            for data in output_file_list:
                csv_writer.writerow(
                    [data.image_file, data.src_file, data.label_idx, data.label_name, data.fold])

    def split_spectrogram(self, img: np.ndarray):
        """
        """
        assert img.shape[0] == self.config.input_shape[0]
        output_width = self.config.input_shape[1]
        stride = int(output_width / 2)

        output_images = []
        for left in range(0, img.shape[1] - output_width + 1, stride):
            right = left + output_width
            output_images.append(img[:, left:right])
        return output_images
