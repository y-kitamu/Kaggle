from pathlib import Path

#DATA_ROOT = Path("/kaggle/input/cassava-leaf-disease-classification/")
DATA_ROOT = Path.home() / "dataset" / "CassavaLeafDisease"

N_CLASSES = 5
BATCH_SIZE = 32
IMAGE_SIZE = 256

TRAIN_CSV = DATA_ROOT / "train.csv"
TRAIN_DATA_DIR = DATA_ROOT / "train_images"
TEST_DATA_DIR = DATA_ROOT / "test_images"

CONFIG_ROOT = (Path(__file__).parents[1] / "confs").resolve()
OUTPUT_ROOT = (Path(__file__).parents[1] / "results").resolve()
