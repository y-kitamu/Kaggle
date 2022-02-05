import enum
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path.home() / "work" / "Kaggle" / "MelanomaClassification"
RESULT_DIR = PROJECT_ROOT / "results"
DATASET_ROOT = Path.home() / "dataset"

# 各 channel ごとの画像の pixel 値の平均と標準偏差 (training image)
IMAGE_MEAN = np.array([83.31454059, 98.79319375, 56.17115773])
IMAGE_STD = np.array([150.63183373, 158.27517638, 206.04443052])


class Labels(enum.Enum):
    benign = 0
    malignant = 1
