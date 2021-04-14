from typing import Optional, List, Tuple

import dataclasses

from official.modeling.hyperparams import base_config
from official.modeling.optimization.configs import optimization_config

OptimizationConfig = optimization_config.OptimizationConfig


@dataclasses.dataclass
class DataConfig(base_config.Config):
    name: Optional[str] = None
    is_training: bool = True
    tfrecords_dir: str = ""
    tfrecords_basename: str = ""


@dataclasses.dataclass
class OptimizerConfig(base_config.Config):
    name: Optional[str] = None
    lr: Optional[float] = None


@dataclasses.dataclass
class ModelConfig(base_config.Config):
    name: Optional[str] = None


@dataclasses.dataclass
class LossConfig(base_config.Config):
    name: Optional[str] = None


@dataclasses.dataclass
class TaskConfig(base_config.Config):
    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    loss: LossConfig = LossConfig()
    train_data: DataConfig = DataConfig()
    validation_data: DataConfig = DataConfig(is_training=False)
    input_shape: Tuple[int, int, int] = (28, 28, 1)
    output_classes: int = 10
    batch_size: int = 4
    steps_per_epoch = 100
    epochs = 20


@dataclasses.dataclass
class TrainerConfig(base_config.Config):
    pass


@dataclasses.dataclass
class ControllerConfig(base_config.Config):
    trainer: TrainerConfig = TrainerConfig()
    task: TaskConfig = TaskConfig()
    strategy: str = "mirrored"
    gpus: Tuple = tuple()
