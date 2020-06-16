from pathlib import Path

import chainer
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from melanoma import constants

DATASET_ROOT = Path.home() / "dataset" / "Melanoma"


class Dataset(chainer.dataset.DatasetMixin):
    """Melanoma dataset
    """
    DATA_ROOT = DATASET_ROOT / "train" / "Normalized"
    METADATAS = ["image_name"]

    def __init__(self, df, n_classes=None, img_size=None, with_metadata=False, is_extend_malignant=True):
        self.df = df
        self.img_size = img_size
        self.n_classes = n_classes
        self.with_metadata = with_metadata

        if is_extend_malignant:
            self._extend_malignant()

        self.n_data = len(self.df)

    def _extend_malignant(self, scale=5):
        malignants = self.df[self.df.benign_malignant == "malignant"]
        self.df = pd.concat([self.df] + [malignants] * scale)

    def get_class_weights(self):
        ratios = np.array([self.n_data / len(self.df[self.df.benign_malignant == l.name]) for l in constants.Labels])
        return ratios / sum(ratios)

    def __len__(self):
        return self.n_data

    def get_example(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(self.DATA_ROOT / f"{row.image_name}.jpg"))
        if self.img_size:
            img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32).transpose(2, 0, 1)
        if self.n_classes:
            label = np.zeros(self.n_classes, dtype=np.int)
            label[constants.Labels[row.benign_malignant].value] = 1
        else:
            label = constants.Labels[row.benign_malignant].value

        if self.with_metadata:
            metadata = {key: row[key] for key in self.METADATAS}
            return img, label, metadata
        else:
            return img, label


class DatasetBuilder:

    def __init__(self,
                 n_classes=2,
                 filename=DATASET_ROOT / "train.csv",
                 train_val_test=(0.8, 0.1, 0.1),
                 random_state=0,
                 img_size=None):
        self.n_classes = n_classes
        self.df = pd.read_csv(filename)
        self.ratios = train_val_test
        self.random_state = random_state
        self.img_size = img_size

    def build(self):
        patients = self.df.patient_id.unique()
        train_val, test = train_test_split(patients,
                                           test_size=self.ratios[2] / sum(self.ratios),
                                           random_state=self.random_state)
        train, val = train_test_split(train_val,
                                      test_size=self.ratios[1] / sum(self.ratios[:2]),
                                      random_state=self.random_state)

        train_dataset = Dataset(self.df[self.df.patient_id.map(lambda x: x in train)],
                                n_classes=self.n_classes,
                                img_size=self.img_size)
        val_dataset = Dataset(self.df[self.df.patient_id.map(lambda x: x in val)],
                              n_classes=self.n_classes,
                              img_size=self.img_size)
        test_dataset = Dataset(self.df[self.df.patient_id.map(lambda x: x in test)],
                               n_classes=self.n_classes,
                               img_size=self.img_size)

        return train_dataset, val_dataset, test_dataset
