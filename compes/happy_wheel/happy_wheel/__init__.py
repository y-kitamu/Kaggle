"""__init__.py
"""
__version__ = "0.1.0"

from pathlib import Path

import ykaggle_core

logger = ykaggle_core.logger

PROJECT_ROOT = Path(__file__).parents[1]
RESULT_ROOT = PROJECT_ROOT / "results"
DATA_ROOT = Path.home() / "dataset" / "HappyWheel"
RAW_TRAIN_DATA_DIR = DATA_ROOT / "train_images"
RAW_TEST_DATA_DIR = DATA_ROOT / "test_images"
TRAIN_CSV = DATA_ROOT / "train.csv"

from . import config, dataset, lr_schedulers, preprocess, train
