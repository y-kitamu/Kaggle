"""__init__.py

Author : Yusuke Kitamura
Create Date : 2022-02-19 13:23:43
Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
"""
import typing
from typing import Any, Callable, Dict

from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras

from .arcface_loss import ArcFaceLoss


def get_loss(loss_name: str, *args, **kwargs) -> keras.losses.Loss:
    """Get loss of `config.loss.name`"""
    mapping: Dict[str, Callable[[Any], keras.losses.Loss]] = {
        "categorical_crossentropy": keras.losses.CategoricalCrossentropy,
        "arcface": ArcFaceLoss,
    }

    if loss_name in mapping:
        return mapping[loss_name](*args, **kwargs)

    raise KeyError(f"No loss_name of '{loss_name}' found in mapping.")
