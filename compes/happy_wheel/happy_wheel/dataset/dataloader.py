"""dataloader.py
"""
import random
from typing import Generator, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from .. import logger
from ..config import DatasetConfig

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
        A.Rotate(limit=5),
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05),
        # A.ImageCompression(quality_lower=85, quality_upper=100, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
        A.HorizontalFlip(),
    ]
)


def build_train_dataloader(
    params: DatasetConfig, is_validation: bool = False
) -> Tuple[tf.data.Dataset, int]:
    """Build train data loader."""
    data_gen = _TrainDataLoader(params)
    path_ds = tf.data.Dataset.from_generator(
        data_gen,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(len(SPECIES)), dtype=tf.int32),
        ),
    )
    ds = path_ds.map(
        lambda path, label: (_load_image(params, path, not is_validation), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if not is_validation:
        ds = ds.repeat()
    return (ds.batch(params.batch_size).prefetch(4), len(data_gen.df))


def build_test_dataloader(params: DatasetConfig) -> tf.data.Dataset:
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
        image = _preprocess_image(image)
    image = _resize_image(image, image_width, image_height)
    tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    tensor = tf.image.resize(tensor, size=[image_height, image_width])
    return tensor


def _preprocess_image(img: np.ndarray) -> np.ndarray:
    """Data augmentation for image data"""
    data = {"image": img}
    aug_data = transforms(**data)
    aug_img = aug_data["image"]
    return aug_img


def _resize_image(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize and padding image."""
    h, w, _ = img.shape
    scale = min(height / h, width / w)
    resized_w = min(int(w * scale), width)
    resized_h = min(int(h * scale), height)
    resized = cv2.resize(img, (resized_w, resized_h))

    target = np.zeros((height, width, 3))
    off_w = random.randint(0, width - resized_w)
    off_h = random.randint(0, height - resized_h)
    target[off_h : resized_h + off_h, off_w : resized_w + off_w] = resized
    return target


class _TrainDataLoader:
    """Generator for training & validataion data.
    Training data must be specified using csv file (`params.label_csv_path`).
    Training csv file has 2 columns : "image" (basename of image file) and "species" (label).

    Args:
        params (DatasetConfig):
    """

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

    def _one_hot(self, index) -> np.ndarray:
        onehot = np.zeros(len(SPECIES), dtype=np.float32)
        onehot[index] = 1.0
        return onehot

    def __call__(self) -> Generator[Tuple[str, np.ndarray], None, None]:
        """Create generator which yields image path and gt label.
        Yields:
            (str, int) : tuple of (image file path, label index).
        """
        df = self.df.sample(len(self.df)) if self.params.shuffle else self.df
        for _, row in df.iterrows():
            yield (
                str(self.params.input_dir / row.image),
                self._one_hot(SPECIES.index(row.species)),
            )


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
