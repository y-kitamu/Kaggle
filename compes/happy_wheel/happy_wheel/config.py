"""config.py
"""
from typing import Any, Dict, Optional

import ykaggle_core as ycore
from pydantic.types import FilePath


class TrainConfig(ycore.config.BaseTrainConfig):
    epochs: int


class ModelConfig(ycore.config.BaseModelConfig):
    kwargs: Dict[Any, Any] = {}
    num_output_class: int


class LossConfig(ycore.config.BaseLossConfig):
    kwargs: Dict[Any, Any] = {}


class OptimizerConfig(ycore.config.BaseOptimizerConfig):
    kwargs: Dict[Any, Any] = {}


class DatasetConfig(ycore.config.BaseImageDatasetConfig):
    label_csv_path: Optional[FilePath] = None


class Config(ycore.config.BaseConfig):
    train: TrainConfig
    model: ModelConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    train_dataset: DatasetConfig

    validation_dataset: DatasetConfig
    test_dataset: DatasetConfig
