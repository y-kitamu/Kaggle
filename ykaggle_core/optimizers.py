"""optimizers.py
"""
import typing
from typing import Any, Callable, Dict

from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras


def get_optimizer(optimizer_name: str, *args, **kwargs) -> keras.optimizers.Optimizer:
    """Get optimizer of `config.optimizer.name`"""
    mapping: Dict[str, Callable[[Any], keras.optimizers.Optimizer]] = {
        "adam": keras.optimizers.Adam
    }

    if optimizer_name in mapping:
        return mapping[optimizer_name](*args, **kwargs)

    raise KeyError(f"No optimizer_name of '{optimizer_name}' found in mapping.")
