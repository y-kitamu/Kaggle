"""config.py
"""
from typing import Optional

import ykaggle_core as ycore
from pydantic.types import FilePath


class TrainConfig(ycore.config.BaseTrainConfig):
    pass


class ModelConfig(ycore.config.BaseModelConfig):
    pass


class LossConfig(ycore.config.BaseLossConfig):
    pass


class OptimizerConfig(ycore.config.BaseOptimizerConfig):
    pass


class DatasetConfig(ycore.config.BaseImageDatasetConfig):
    label_csv_path: Optional[FilePath] = None


class Config(ycore.config.BaseConfig):
    train: TrainConfig
    model: ModelConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    dataset: DatasetConfig
