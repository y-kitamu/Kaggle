"""dataset.py
"""
import random
from pathlib import Path
from typing import Generator, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from . import logger
from .config import DatasetConfig

SPECIES = [
    "beluga",
    "blue_whale",
    "bottlenose_dolphin",
    "bottlenose_dolpin",
    "brydes_whale",
    "commersons_dolphin",
    "common_dolphin",
    "cuviers_beaked_whale",
    "dusky_dolphin",
    "false_killer_whale",
    "fin_whale",
    "frasiers_dolphin",
    "globis",
    "gray_whale",
    "humpback_whale",
    "kiler_whale",
    "killer_whale",
    "long_finned_pilot_whale",
    "melon_headed_whale",
    "minke_whale",
    "pantropic_spotted_dolphin",
    "pilot_whale",
    "pygmy_killer_whale",
    "rough_toothed_dolphin",
    "sei_whale",
    "short_finned_pilot_whale",
    "southern_right_whale",
    "spinner_dolphin",
    "spotted_dolphin",
    "white_sided_dolphin",
]

transforms = A.Compose(
    [
        A.Rotate(limit=40),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2),
        A.ImageCompression(quality_lower=85, quality_upper=100, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.HorizontalFlip(),
    ]
)


def build_train_dataloader(params: DatasetConfig, is_validation: bool = False):
    """Build train data loader."""
    data_gen = _TrainDataLoader(params)
    path_ds = tf.data.Dataset.from_generator(
        data_gen,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    ds = path_ds.map(
        lambda path, label: (_load_image(params, path, not is_validation), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if not is_validation:
        ds = ds.repeat()
    return ds.batch(params.batch_size).prefetch(4)


def build_test_dataloader(params: DatasetConfig):
    """Build test data loader."""
    data_gen = _TestDataLoader(params)
    path_ds = tf.data.Dataset.from_generator(
        data_gen, output_signature=(tf.TensorSpec(shape=(), dtype=tf.string))
    )
    ds = (
        path_ds.map(
            lambda path: _load_image(params, path, False), num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(params.batch_size)
        .prefetch(4)
    )
    return ds


def _load_image(params: DatasetConfig, img_path: tf.string, with_preprocess: bool) -> tf.Tensor:
    aug_img = tf.numpy_function(
        func=_load_image_impl,
        inp=[img_path, with_preprocess, params.width, params.height],
        Tout=tf.float32,
    )
    return aug_img


def _load_image_impl(
    img_path: bytes, with_preprocess: bool, image_width: int, image_height: int
) -> tf.Tensor:
    """load image"""
    image = cv2.imread(img_path.decode())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if with_preprocess:
        image = _preprocess_image(image, image_height, image_width)
    return image


def _preprocess_image(img: np.ndarray, image_height: int, image_width: int) -> tf.Tensor:
    """Data augmentation for image data"""
    data = {"image": img}
    aug_data = transforms(**data)
    aug_img = aug_data["image"]
    aug_img = img
    aug_img = tf.convert_to_tensor(aug_img / 255.0, dtype=tf.float32)
    aug_img = tf.image.resize(aug_img, size=[image_height, image_width])
    return aug_img


class _TrainDataLoader:
    """Generator for training & validataion data."""

    def __init__(self, params: DatasetConfig):
        if params.label_csv_path is None:
            raise Exception("params.label_csv_path must not be None.")
        self.params = params
        self.images = list(self.params.input_dir.glob("**/*.jpg"))
        self.df: pd.DataFrame = pd.read_csv(params.label_csv_path)
        # self.df = self._validate_csv()

    def _validate_csv(self) -> pd.DataFrame:
        images = [f.name for f in self.images]
        is_valid = self.df.image.map(lambda x: x in images)
        logger.info(
            f"Validation result of image data in csv file : {sum(is_valid)} / {len(is_valid)}"
        )
        return self.df[is_valid]

    def __call__(self) -> Generator[Tuple[str, int], None, None]:
        """Create generator which yields image path and gt label.
        Yields:
            (str, int) : tuple of (image file path, label index).
        """
        df = self.df.sample(len(self.df)) if self.params.shuffle else self.df
        for _, row in df.iterrows():
            yield (str(self.params.input_dir / row.image), SPECIES.index(row.species))


class _TestDataLoader:
    def __init__(self, params: DatasetConfig):
        self.params = params
        self.images = list(self.params.input_dir.glob("**/*.jpg"))

    def __call__(self) -> Generator[str, None, None]:
        """Create generator which yields image path.
        Yields:
             (str) : image file path.
        """
        images = (
            random.sample(self.images, len(self.images)) if self.params.shuffle else self.images
        )
        for img_path in images:
            yield str(img_path)
