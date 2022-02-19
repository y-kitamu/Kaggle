"""__init__.py
"""
__version__ = "0.1.0"

from pathlib import Path

import ykaggle_core

from .identities import IDENTITIES

logger = ykaggle_core.logger

# constants
PROJECT_ROOT = Path(__file__).parents[1]
RESULT_ROOT = PROJECT_ROOT / "results"
DATA_ROOT = Path.home() / "dataset" / "HappyWheel"
RAW_TRAIN_DATA_DIR = DATA_ROOT / "train_images"
RAW_TEST_DATA_DIR = DATA_ROOT / "test_images"
TRAIN_CSV = DATA_ROOT / "train.csv"

NUM_INDIVIDUALS = len(IDENTITIES)

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

from . import config, dataset, lr_schedulers, preprocess, train
