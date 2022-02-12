"""loss.py
"""
import typing
from typing import Any, Callable, Dict

from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras


def get_loss(loss_name: str, *args, **kwargs) -> keras.losses.Loss:
    """Get loss of `config.loss.name`"""
    mapping: Dict[str, Callable[[Any], keras.losses.Loss]] = {
        "categorical_crossentropy": keras.losses.CategoricalCrossentropy
    }

    if loss_name in mapping:
        return mapping[loss_name](*args, **kwargs)

    raise KeyError(f"No loss_name of '{loss_name}' found in mapping.")
