from typing import Optional, Tuple

import dataclasses


@dataclasses.dataclass
class DataConfig:
    name: str = ""
    is_training: bool = True
    num_data: int = 60000
    tfrecords_dir: str = ""
    tfrecords_basename: str = ""


@dataclasses.dataclass
class OptimizerConfig:
    name: Optional[str] = None
    lr: Optional[float] = None


@dataclasses.dataclass
class ModelConfig:
    name: Optional[str] = None


@dataclasses.dataclass
class LossConfig:
    name: Optional[str] = None


@dataclasses.dataclass
class TaskConfig:
    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    loss: LossConfig = LossConfig()
    train_data: DataConfig = DataConfig()
    validation_data: DataConfig = DataConfig(is_training=False)
    input_shape: Tuple[int, int, int] = (28, 28, 1)
    output_classes: int = 10
    batch_size: int = 4
    steps_per_epoch: int = 100
    epochs: int = 20


@dataclasses.dataclass
class TrainerConfig:
    pass


@dataclasses.dataclass
class ControllerConfig:
    trainer: TrainerConfig = TrainerConfig()
    task: TaskConfig = TaskConfig()
    strategy: str = "mirrored"
    gpus: Tuple = tuple()
