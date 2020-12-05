"""Dataloader for training, validation, test
"""
import os
import glob
import math
from queue import Queue
from multiprocessing.pool import Pool

import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.constant import DATA_ROOT, N_CLASSES, BATCH_SIZE, IMAGE_SIZE

# data loader
TRAIN_CSV = DATA_ROOT / "train.csv"
TRAIN_DATA_DIR = DATA_ROOT / "train_images"
TEST_DATA_DIR = DATA_ROOT / "test_images"


def scale(affine_mat, scale_x, scale_y=None):
    scale_y = scale_y or scale_x
    return np.matmul(np.array([[scale_x, 0], [0, scale_y]]), affine_mat)


def horizontal_flip(affine_mat, width):
    return np.matmul(np.array([[-1, 0], [0, 1]]), affine_mat) + np.array([[0, 0, width], [0, 0, 0]])


def augment(affine_mat, image_size):
    # horizontal flip
    if np.random.rand() > 0.5:
        affine_mat = horizontal_flip(affine_mat, image_size)
    return affine_mat


def preprocess(filename, image_size, is_train):
    image = cv2.imread(os.path.join(filename)) / 255.0
    affine_mat = np.array([[1, 0, 0], [0, 1, 0]])
    affine_mat = scale(affine_mat, image_size / image.shape[1], image_size / image.shape[0])
    # if is_train:
    #     affine_mat = augment(affine_mat, image_size)
    image = cv2.warpAffine(image, affine_mat, (image_size, image_size))
    return image


def fetch(filenames, labels, data_dir, image_size, is_train):
    """Create batch data. Read and augment images.
    Args:
        filenames (list of str) : list of image filenames
    """
    images = np.zeros((len(filenames), image_size, image_size, 3))
    for idx, fname in enumerate(filenames):
        filename = os.path.join(data_dir, fname)
        image = preprocess(filename, image_size, is_train)
        images[idx, ...] = image
    return images, labels


class DatasetGenerator:
    """
    """

    def __init__(self,
                 df,
                 is_train=True,
                 data_dir=str(TRAIN_DATA_DIR),
                 n_classes=N_CLASSES,
                 image_size=IMAGE_SIZE,
                 batch_size=BATCH_SIZE,
                 n_prefetch=4):
        self.filenames = df.image_id.to_numpy()
        self.labels = np.identity(n_classes)[df.label.to_numpy().astype(np.int)]
        self.data_dir = data_dir
        self.is_train = is_train
        self.n_classes = n_classes
        self.image_size = image_size
        self.batch_size = batch_size
        self.samples = len(df)

        # for prefetch
        self.process_pool = Pool(processes=n_prefetch)
        self.queue = Queue(maxsize=n_prefetch)
        self.files_and_labels_gen = self._get_files_and_labels_generator()

    def __len__(self):
        return len(self.filenames)

    def __iter__(self):
        self.files_and_labels_gen = self._get_files_and_labels_generator()
        return self

    def __next__(self):
        self._prefetch()
        return self.queue.get().get()

    def _prefetch(self):
        while not self.queue.full():
            filenames, labels = next(self.files_and_labels_gen)
            self.queue.put(
                self.process_pool.apply_async(
                    fetch, (filenames, labels, self.data_dir, self.image_size, self.is_train)))

    def _get_files_and_labels_generator(self):
        steps_per_epoch = math.ceil(self.samples / self.batch_size)
        if self.is_train:
            while True:
                index_arr = np.arange(steps_per_epoch * self.batch_size, dtype=np.int)
                index_arr[self.samples:] = np.random.randint(self.samples)
                shuffled = np.random.permutation(index_arr)
                for i in range(steps_per_epoch):
                    start = i * self.batch_size
                    end = start + self.batch_size
                    yield self.filenames[shuffled[start:end]], self.labels[shuffled[start:end]]
        else:
            while True:
                for i in range(steps_per_epoch):
                    start = i * self.batch_size
                    end = min(start + self.batch_size, len(self.filenames))
                    yield self.filenames[start:end], self.labels[start:end]


class TestDatasetGenerator:

    def __init__(self,
                 directory,
                 n_classes=N_CLASSES,
                 image_size=IMAGE_SIZE,
                 batch_size=BATCH_SIZE,
                 n_prefetch=4):
        self.filenames = sorted(
            [os.path.basename(fname) for fname in glob.glob(os.path.join(directory, "*.jpg"))])
        self.data_dir = directory
        self.n_classes = n_classes
        self.image_size = image_size
        self.batch_size = batch_size
        self.samples = len(self.filenames)

        # for prefetch
        self.process_pool = Pool(processes=n_prefetch)
        self.queue = Queue(maxsize=n_prefetch)
        self.files_and_labels_gen = self._get_files_and_labels_generator()

    def __len__(self):
        return len(self.filenames)

    def __iter__(self):
        self.files_and_labels_gen = self._get_files_and_labels_generator()
        return self

    def __next__(self):
        self._prefetch()
        return self.queue.get().get()[0]

    def _prefetch(self):
        while not self.queue.full():
            filenames, labels = next(self.files_and_labels_gen)
            if filenames is None:
                break
            self.queue.put(
                self.process_pool.apply_async(
                    fetch, (filenames, labels, self.data_dir, self.image_size, False)))

    def _get_files_and_labels_generator(self):
        steps_per_epoch = math.ceil(self.samples / self.batch_size)
        for i in range(steps_per_epoch):
            start = i * self.batch_size
            end = min(start + self.batch_size, len(self.filenames))
            yield self.filenames[start:end], None
        yield None, None


def get_train_val_dataset(cfg, test_ratio=0.2):
    n_classes = cfg.n_classes
    df = pd.read_csv(TRAIN_CSV)
    train_df, val_df = train_test_split(df, test_size=test_ratio, stratify=df.label)
    train_gen = DatasetGenerator(
        train_df,
        is_train=True,
        n_classes=n_classes,
        batch_size=cfg.train.batch_size,
        image_size=cfg.image_size,
    )

    val_gen = DatasetGenerator(
        val_df,
        is_train=False,
        n_classes=n_classes,
        batch_size=cfg.train.batch_size * 2,
        image_size=cfg.image_size,
    )
    return train_gen, val_gen


def get_kfold_dataset(cfg):
    df = pd.read_csv(TRAIN_CSV)
    kf = StratifiedKFold(n_splits=cfg.train.k_fold, shuffle=True)
    for train_idx, val_idx in kf.split(df.image_id, df.label):
        train_gen = DatasetGenerator(df.iloc[train_idx],
                                     is_train=True,
                                     n_classes=cfg.n_classes,
                                     batch_size=cfg.train.batch_size,
                                     image_size=cfg.image_size)
        val_gen = DatasetGenerator(df.iloc[val_idx],
                                   is_train=False,
                                   n_classes=cfg.n_classes,
                                   batch_size=cfg.train.batch_size * 2,
                                   image_size=cfg.image_size)
        yield train_gen, val_gen
