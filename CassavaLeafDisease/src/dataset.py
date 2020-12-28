"""gDataloader for training, validation, test
"""
import os
import math
import glob
import random
from queue import Queue
from multiprocessing.pool import Pool

import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.constant import DATA_ROOT, N_CLASSES, BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT

# data loader
TRAIN_CSV = DATA_ROOT / "train.csv"
TRAIN_DATA_DIR = DATA_ROOT / "train_images"
TEST_DATA_DIR = DATA_ROOT / "test_images"


def dense_to_onehot(array, n_classes=5):
    """Convert categorical label to onehot label.
    Args:
        array (np.ndarray) : If array's dimension is 1, convert to one_hot style.
           Else if the dimension is 2 or array is None, return as is.
           Otherwise, raise ValueError.
        n_classes (int)    :
    Return:
        np.ndarray : one-hot encoded label array (2D array)
    """
    if array is None or len(array.shape) == 2:
        return array
    if len(array.shape) == 1:
        return np.identity(n_classes)[array]
    raise ValueError("Input array has invalid shape : {}".format(array.shape))


def scale(affine_mat, scale_x, scale_y=None):
    scale_y = scale_y or scale_x
    return np.matmul(np.array([[scale_x, 0], [0, scale_y]]), affine_mat)


def random_shift(affine_mat, max_shift_x, max_shift_y):
    shift_x = np.clip(np.random.normal(loc=0.5, scale=0.16), 0, 1) * max_shift_x
    shift_y = np.clip(np.random.normal(loc=0.5, scale=0.16), 0, 1) * max_shift_y
    affine_mat[0, 2] -= shift_x
    affine_mat[1, 2] -= shift_y
    return affine_mat


def rotate(affine_mat, angle, center_x, center_y):
    rot = cv2.getRotationMatrix2D((center_x, center_y), angle, scale=1.0)
    affine_mat = np.dot(rot[:, :2], affine_mat)
    affine_mat[:, 2] += rot[:, 2]
    return affine_mat


def horizontal_flip(affine_mat, width):
    return np.matmul(np.array([[-1, 0], [0, 1]]), affine_mat) + np.array([[0, 0, width], [0, 0, 0]])


def augment(affine_mat, image_width, image_height, max_angle=30):
    # random flip (horizontal)
    if np.random.rand() > 0.5:
        affine_mat = horizontal_flip(affine_mat, image_width)

    # random rotate
    angle = 2 * max_angle * np.random.rand() - max_angle
    rotate(affine_mat, angle, image_width / 2, image_height / 2)

    return affine_mat


def preprocess(filename, image_width, image_height, is_train):
    """Load image and apply augment.
    At first image is scaled from (original width, original height) to (image_width, image_height).
    Then, augmentation is applied to the image.

    Args:
        filename (str)   : image filename to be loaded.
        image_width (int) : output image width
        image_height (int) : output image height
        is_train (bool)  : If True, augmentation is applied to the image,
            else no augmentation is applied.
    Return:
        np.ndarray : image data array of shape [image_width, image_height, channel]
    """
    image = cv2.imread(os.path.join(filename)) / 255.0
    affine_mat = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    if is_train:
        affine_mat = augment(affine_mat, image.shape[1], image.shape[0])
    affine_mat = random_shift(affine_mat, max(0, image.shape[1] - image_width),
                              max(0, image.shape[0] - image_height))
    image = cv2.warpAffine(image, affine_mat, (image_width, image_height))

    if is_train:
        image += np.random.rand(*image.shape) * 0.1 - 0.05
        image = np.clip(image, 0.0, 1.0)
    return image.astype(np.float32)


def fetch(filenames, labels, data_dir, image_width, image_height, is_train):
    """Create batch data. Read and augment images.
    Args:
        filenames (list of str) : list of image filenames (basenames)
        labels (np.ndarray)     : label array
        data_dir (str)          : Path to the directory where image files exis.
        image_width (int)        : output image width
        image_height (int)        : output image height
        is_train (bool)         : If True, augmentation is applied to images.
    Return:
        np.ndarray : image array of [Batch size, image_height, image_width, channel]
        np.ndarray : label array (one-hot or categorical encoded).
            If `labels` is None, This variable is not returned.
    """
    images = np.zeros((len(filenames), image_height, image_width, 3))
    for idx, fname in enumerate(filenames):
        filename = os.path.join(data_dir, fname)
        image = preprocess(filename, image_width, image_height, is_train)
        images[idx, ...] = image
    if labels is None:  # if test
        return images
    return images, labels


class BaseDatasetGenerator:
    """Base dataloader.

    Args:
        filenames (list of str) : list of input image filenames (basenames)
        labels (np.ndarray)     : True label's array correspond to `filenames`
        data_dir (str)          : Path to the directory where input images exist.
        n_classes (int)         :
        image_width (int)       : output image size
        image_height (int)      : output image size
        batch_size (int)        :
        n_prefetch (int)        : Number of prefetched batches.
    """

    def __init__(self,
                 filenames,
                 labels=None,
                 data_dir=str(TRAIN_DATA_DIR),
                 n_classes=N_CLASSES,
                 image_width=IMAGE_WIDTH,
                 image_height=IMAGE_HEIGHT,
                 batch_size=BATCH_SIZE,
                 n_prefetch=4):
        self.filenames = filenames
        self.labels = dense_to_onehot(labels, n_classes)
        if self.labels is not None:
            assert len(self.filenames) == len(self.labels)
        self.data_dir = data_dir
        self.n_classes = n_classes
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        self.samples = len(self.filenames)
        self._is_train = None

        # for prefetch
        self.process_pool = Pool(processes=n_prefetch)
        self.queue = Queue(maxsize=n_prefetch)
        self.files_and_labels_gen = self._get_files_and_labels_generator()

    @property
    def is_train(self):
        if self._is_train is None:
            raise ValueError(
                "`self._is_train` must be True or False, not None. You should set the value in the constructor of the inheritance of BaseDatasetGenerator."
            )
        return self._is_train

    def __len__(self):
        return len(self.filenames)

    def __iter__(self):
        while not self.queue.empty():
            self.queue.get()
        self.files_and_labels_gen = self._get_files_and_labels_generator()
        return self

    def __del__(self):
        if hasattr(self, "process_pool"):
            self.process_pool.terminate()

    def __next__(self):
        self._prefetch()
        if self.queue.empty():
            self.files_and_labels_gen = self._get_files_and_labels_generator()
            raise StopIteration
        return self.queue.get().get()

    def _prefetch(self):
        while not self.queue.full():
            filenames, labels = next(self.files_and_labels_gen)
            if filenames is None:
                break
            self.queue.put(
                self.process_pool.apply_async(fetch, (filenames, labels, self.data_dir, self.image_width,
                                                      self.image_height, self.is_train)))

    def _get_files_and_labels_generator(self):
        raise NotImplementedError("Function _get_files_and_labels_generator is not implemented.")


class TrainDatasetGenerator(BaseDatasetGenerator):
    """
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_train = True

    def _get_files_and_labels_generator(self):
        steps_per_epoch = math.ceil(self.samples / self.batch_size)
        while True:
            index_arr = np.arange(steps_per_epoch * self.batch_size, dtype=np.int)
            index_arr[self.samples:] = np.random.randint(index_arr.shape[0] - self.samples)
            shuffled = np.random.permutation(index_arr)
            for i in range(steps_per_epoch):
                start = i * self.batch_size
                end = start + self.batch_size
                yield self.filenames[shuffled[start:end]], self.labels[shuffled[start:end]]


class TestDatasetGenerator(BaseDatasetGenerator):
    """
    """

    def __init__(self, *args, with_label=True, repeat=False, **kwargs):
        if repeat:
            self._get_files_and_labels_generator = self._repeat_generator
        else:
            self._get_files_and_labels_generator = self._generator
        super().__init__(*args, **kwargs)
        self._is_train = False
        self.with_label = self.labels is not None and with_label

    def repeat(self, is_repeat):
        self.process_pool = Pool(processes=4)
        self.queue = Queue(maxsize=4)
        if is_repeat:
            self._get_files_and_labels_generator = self._repeat_generator
        else:
            self._get_files_and_labels_generator = self._generator

    def _repeat_generator(self):
        while True:
            for img, label in self._generator():
                yield img, label

    def _generator(self):
        steps_per_epoch = math.ceil(self.samples / self.batch_size)
        for i in range(steps_per_epoch):
            start = i * self.batch_size
            end = min(start + self.batch_size, len(self.filenames))
            if self.with_label:
                yield self.filenames[start:end], self.labels[start:end]
            else:
                yield self.filenames[start:end], None


def get_train_val_dataset(cfg,
                          data_dir=TRAIN_DATA_DIR,
                          csv_fname=TRAIN_CSV,
                          test_ratio=0.2,
                          is_train=False):
    """Read csv, split datas to train and val and get dataset generator
    Args:
        cfg (Omegaconf.DefaultDict) : configurations imported by hydra.
        data_dir (str) : Path to the directory where input images exist.
        csv_fname (str) : Path to the csv file in which image filenames and true labels are listed.
        test_ratio (float) : test data ratio. (Only used when is_train == True)
        is_train (bool) : If True, dataset is split into train and validation set.
    Return:
        BaseDatasetGenerator : Training dataset generator
        BaseDatasetGenerator : If `is_train` is False, this variable is None.
            Else, validation dataset's generator is returned.
    """
    n_classes = cfg.n_classes
    df = pd.read_csv(csv_fname)
    if not is_train:
        gen = TestDatasetGenerator(
            df.image_id.to_numpy(),
            labels=df.label.to_numpy().astype(np.int),
            data_dir=data_dir,
            n_classes=n_classes,
            batch_size=cfg.train.batch_size,
            image_width=cfg.image_width,
            image_height=cfg.image_height,
        )
        return gen, None

    train_df, val_df = train_test_split(df, test_size=test_ratio, stratify=df.label)
    train_gen = TrainDatasetGenerator(
        train_df.image_id.to_numpy(),
        train_df.label.to_numpy().astype(np.int),
        data_dir=data_dir,
        n_classes=n_classes,
        batch_size=cfg.train.batch_size,
        image_width=cfg.image_width,
        image_height=cfg.image_height,
    )

    val_gen = TestDatasetGenerator(
        val_df.image_id.to_numpy(),
        df.label.to_numpy().astype(np.int),
        data_dir=data_dir,
        n_classes=n_classes,
        batch_size=cfg.train.batch_size * 2,
        image_width=cfg.image_width,
        image_height=cfg.image_height,
    )
    return train_gen, val_gen


def get_kfold_dataset(cfg):
    """Return generator of datasets for k-fold cross validation.
    Args:
        cfg (OmegaConf.DefaultDict) : configurations
    Return:
        generator : generator that yields
            (BaseDatasetGenerator, BaseDatasetGenerator) for `cfg.train.k_fold` times.
    """
    df = pd.read_csv(TRAIN_CSV)
    kf = StratifiedKFold(n_splits=cfg.train.k_fold, shuffle=True)
    for train_idx, val_idx in kf.split(df.image_id, df.label):
        train_df = df.iloc[train_idx]
        train_gen = TrainDatasetGenerator(
            train_df.image_id.to_numpy(),
            train_df.label.to_numpy(),
            n_classes=cfg.n_classes,
            batch_size=cfg.train.batch_size,
            image_width=cfg.image_width,
            image_height=cfg.image_height,
        )
        val_df = df.iloc[val_idx]
        val_gen = TestDatasetGenerator(
            val_df.image_id.to_numpy(),
            val_df.label.to_numpy(),
            n_classes=cfg.n_classes,
            batch_size=cfg.train.batch_size * 2,
            repeat=True,
            image_width=cfg.image_width,
            image_height=cfg.image_height,
        )
        yield train_gen, val_gen


def get_test_dataset(test_data_dir="../input/cassava-leaf-disease-classification"):
    """Get dataset for test.
    Args:
        test_data_dir (str) : Path to the directory where input images exist.
    Return:
        TestDatasetGenerator : test dataset generator
    """
    file_list = [os.path.basename(fname) for fname in glob.glob(os.path.join(test_data_dir, "*"))]
    test_ds = TestDatasetGenerator(file_list, with_label=False)
    return test_ds
