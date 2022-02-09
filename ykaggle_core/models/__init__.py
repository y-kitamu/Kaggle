"""__init__.py
"""
import typing
from typing import Any, Callable, Dict

from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras


def get_model(model_name: str, *args, **kwargs) -> keras.Model:
    mapping: Dict[str, Callable[[Any], keras.Model]] = {
        "efficientnet-b0": keras.applications.EfficientNetB0,
    }
    if model_name in mapping:
        return mapping["model_name"](*args, **kwargs)
    raise KeyError(f"No model_name of '{model_name}' found in mapping.")
