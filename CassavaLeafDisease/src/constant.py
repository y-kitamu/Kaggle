from pathlib import Path
from dataclasses import dataclass

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


@dataclass
class PartsParams:
    class_name: str
    config: dict = {}


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
    image_size: int = 256
    n_channel: int = 3
    train: TrainParams = TrainParams()
