import random
from pathlib import Path

import chainer
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from melanoma import constants
from melanoma import imageproc

DATASET_ROOT = Path.home() / "dataset" / "Melanoma"


class Dataset(chainer.dataset.DatasetMixin):
    """Melanoma dataset
    Args:
        df (pandas.DataFrame) : dataset
        n_classes (int)
        img_size (tuple or list): (width, height)
        with_metadata (Bool) : If True, `get_example` returns with metadata.
        is_extend_malignant (Bool) : If True, return,
        preprocess (list of function or function) : preprocess (augmentation) image function object
    """
    DATA_ROOT = DATASET_ROOT / "train" / "Resized"
    METADATAS = ["image_name"]

    def __init__(self,
                 df,
                 n_classes=None,
                 img_size=None,
                 with_metadata=False,
                 is_extend_malignant=True,
                 preprocess=[imageproc.normalize]):
        self.df = df
        self.img_size = img_size
        self.n_classes = n_classes
        self.with_metadata = with_metadata
        self.preprocess = preprocess

        if is_extend_malignant:
            self._extend_malignant()

        self.n_data = len(self.df)

    def _extend_malignant(self, scale=None):
        # malignants = self.df[self.df.benign_malignant == "malignant"]
        # scale = scale or int(0.5 * len(self.df) / len(malignants))
        # self.df = pd.concat([self.df] + [malignants] * scale)
        df_0 = self.df[self.df.target == constants.Labels.benign.value]
        df_1 = self.df[self.df.target == constants.Labels.malignant.value]
        sample = min(len(df_1) * 10, len(df_0))
        df_0 = df_0.sample(sample, random_state=0)
        self.df = pd.concat([df_0, df_1])

    def get_class_weights(self):
        ratios = np.array([self.n_data / len(self.df[self.df.target == l.value]) for l in constants.Labels])
        return ratios / sum(ratios)

    def __len__(self):
        return self.n_data

    def _read_img(self, img_filename):
        img = cv2.imread(img_filename)
        if self.img_size:
            img = cv2.resize(img, self.img_size)
        if self.preprocess is not None:
            if isinstance(self.preprocess, (list, tuple)):
                for pp in self.preprocess:
                    img = pp(img)
            else:
                img = self.preprocess(img)
        img = img.astype(np.float32).transpose(2, 0, 1)
        return img

    def get_example(self, idx):
        row = self.df.iloc[idx]
        img = self._read_img(str(self.DATA_ROOT / f"{row.image_name}.png"))
        if self.n_classes:
            label = np.zeros(self.n_classes, dtype=np.int)
            label[row.target] = 1
        else:
            label = row.target

        if self.with_metadata:
            metadata = {key: row[key] for key in self.METADATAS}
            return img, label, metadata
        else:
            return img, label


class SubmissionDataset(Dataset):
    DATA_ROOT = DATASET_ROOT / "test" / "Resized"

    def __init__(self, filename, img_size=None):
        super().__init__(pd.read_csv(filename), img_size, is_extend_malignant=False)

    def get_example(self, idx):
        row = self.df.iloc[idx]
        return self._read_img(str(self.DATA_ROOT / f"{row.image_name}.png")), row.image_name


class DatasetBuilder:

    def __init__(self,
                 n_classes=2,
                 filename=DATASET_ROOT / "train.csv",
                 train_val_test=(0.8, 0.1, 0.1),
                 random_state=0,
                 img_size=None,
                 preprocess=[imageproc.normalize, imageproc.standard_augmentation],
                 **kwargs):
        """
        Args:
            filename (pathlib.Path) : csv filename
            train_val_test (tuple of float): ratio of training, validation, test data
            img_size (tuple or list): (width, height)
            **kwargs : kwargs that pass to Dataset
        """
        self.n_classes = n_classes
        self.df = pd.read_csv(filename)
        self.ratios = train_val_test
        self.random_state = random_state
        self.img_size = img_size
        self.preprocess = preprocess
        self.kwargs = kwargs

    def build(self):
        patients = self.df.patient_id.unique()
        train_val, test = train_test_split(patients,
                                           test_size=self.ratios[2] / sum(self.ratios),
                                           random_state=self.random_state)
        train, val = train_test_split(train_val,
                                      test_size=self.ratios[1] / sum(self.ratios[:2]),
                                      random_state=self.random_state)
        return self._build(train, val, test)

    def _build(self, train, val, test):
        """
        Args:
            train (list of string) : train data's patient id list.
            val (list of string) : validation data's patient id list.
            test (list of string) : test data's patient id list.
        """
        train_dataset = Dataset(self.df[self.df.patient_id.map(lambda x: x in train)],
                                n_classes=self.n_classes,
                                img_size=self.img_size,
                                **self.kwargs)
        val_dataset = Dataset(self.df[self.df.patient_id.map(lambda x: x in val)],
                              n_classes=self.n_classes,
                              img_size=self.img_size,
                              **self.kwargs)
        test_dataset = Dataset(self.df[self.df.patient_id.map(lambda x: x in test)],
                               n_classes=self.n_classes,
                               img_size=self.img_size,
                               is_extend_malignant=False)
        train_dataset.preprocess = self.preprocess
        val_dataset.preprocess = self.preprocess

        return train_dataset, val_dataset, test_dataset

    def get_cross_validation_dataset_generator(self, n_folds=3):
        patients = list(self.df.patient_id.unique())
        random.seed(self.random_state)
        random.shuffle(patients)
        folds = np.array_split(patients, n_folds)
        for i in range(n_folds):
            train_val = list(set(patients) - set(folds[i]))
            train, val = train_test_split(train_val,
                                          test_size=self.ratios[1] / sum(self.ratios[:2]),
                                          random_state=self.random_state)
            yield self._build(train, val, folds[i])
