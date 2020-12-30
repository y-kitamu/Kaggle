import os
from pathlib import Path
from dataclasses import dataclass, field

DATA_ROOT = Path.home() / "dataset" / "CassavaLeafDisease"
# for kaggle notebook
if os.path.exists("/kaggle/input/cassava-leaf-disease-classification/"):
    DATA_ROOT = Path("/kaggle/input/cassava-leaf-disease-classification/")
# for google colab
if os.path.exists("/content"):
    DATA_ROOT = Path("/content")

N_CLASSES = 5
BATCH_SIZE = 32
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600

TRAIN_CSV = DATA_ROOT / "train.csv"
TRAIN_DATA_DIR = DATA_ROOT / "train_images"
TEST_DATA_DIR = DATA_ROOT / "test_images"

CONFIG_ROOT = (Path(__file__).parents[0] / "confs").resolve()
OUTPUT_ROOT = (Path(__file__).parents[1] / "results").resolve()
# for kaggle notebook
if os.path.exists("/kaggle/input"):
    OUTPUT_ROOT = Path("/kaggle/working/results/")
# for google colab
if os.path.exists("/content/gdrive"):
    OUTPUT_ROOT = Path("/content/gdrive/MyDrive/Colab Notebooks/Kaggle/CassaveLeafDisease/result")


@dataclass
class PartsParams:
    class_name: str
    config: dict = field(default_factory=dict)


@dataclass
class TrainParams:
    k_fold: int = 5
    epochs: int = 95
    batch_size: int = 32
    start_epoch: int = 0
    initial_lr: float = 0.001
    model: PartsParams = PartsParams(class_name="efficientnetb0", config={})
    optimizer: PartsParams = PartsParams(class_name="Adam", config={"leraning_rate": initial_lr})
    loss: PartsParams = PartsParams(class_name="CategoricalCrossentropy",
                                    config={
                                        "from_logits": True,
                                        "label_smoothing": 0.1
                                    })
    lr_schedule: PartsParams = PartsParams(class_name="manual_lr_scheduler",
                                           config={
                                               "warmup_epoch": 5,
                                               "annealing_epoch": 30,
                                               "annealing_scale": 0.1,
                                               "num_annealing_step": 5
                                           })


@dataclass
class Params:
    title: str
    gpu: int = 0
    n_classes: int = 5
    image_width: int = 512
    image_height: int = 512
    n_channel: int = 3
    train: TrainParams = TrainParams()
