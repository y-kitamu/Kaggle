"""config.py
"""
from pydantic import BaseModel
from pydantic.types import DirectoryPath


class BaseTrainConfig(BaseModel):
    image_width: int
    image_height: int


class BaseModelConfig(BaseModel):
    name: str


class BaseLossConfig(BaseModel):
    name: str


class BaseOptimizerConfig(BaseModel):
    name: str


class BaseImageDatasetConfig(BaseModel):
    input_dir: DirectoryPath
    batch_size: int
    width: int
    height: int
    shuffle: bool


class BaseConfig(BaseModel):
    train: BaseTrainConfig
    model: BaseModelConfig
    loss: BaseLossConfig
    optimizer: BaseOptimizerConfig
    dataset: BaseImageDatasetConfig
