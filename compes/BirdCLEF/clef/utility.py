import os
import pdb
import traceback
import sys
import logging
import multiprocessing
from typing import Union, List, Callable, Any

import tensorflow as tf
from numba import cuda

import clef


def run_debug(func: Callable) -> Any:
    """Start pdb debugger at where the `func` throw Exception.
    """
    try:
        res = func()
        return res
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


def clear_gpu(gpu_id: Union[int, List[int]] = 0) -> None:
    """Release gpu memory
    Args:
        gpu_id (int) : GPU of which memory is released.
    """
    if isinstance(gpu_id, int):
        gpu_id = [gpu_id]
    for id in gpu_id:
        cuda.select_device(id)
        device = cuda.get_current_device()
        device.reset()
        cuda.close()
        clef.logger.info("CUDA memory released: GPU {}".format(id))


class LogLevel():
    """指定したloggerのloglevelを指定した値に一時的に変更する
    Usage:
    ```python
    with LogLevel(10, logger):
        <some operations>
    ```

    Args:
        loglevel (int) : 設定するloglevel
        logger (logging.Logger) : `loglevel`を適用するlogger
    """

    def __init__(self, loglevel: int, logger: logging.Logger = clef.logger):
        self.loglevel = loglevel
        self.logger = logger

    def __enter__(self):
        self.default_level = self.logger.level
        self.logger.setLevel(self.loglevel)

    def __exit__(self, ex_type, ex_value, trace):
        self.logger.setLevel(self.default_level)


class GPUInitializer:
    """使用するGPUの設定、初期化
    """

    def __init__(self) -> None:
        self.is_gpu_initialized = False

    def __call__(self, gpu_id=0, overwrite=False) -> None:
        """初期化（memory release）、GPUの設定
        """
        if self.is_gpu_initialized and not overwrite:
            clef.logger.warning("GPU is already initialized. Skip set gpu.")
            return
        if isinstance(gpu_id, int):
            gpu_id = [gpu_id]
        if len(gpu_id) == 0 or gpu_id[0] < 0:
            tf.config.set_visible_devices([], 'GPU')
            return
        _set_gpu_impl(gpu_id)
        self.is_gpu_initialized = True


set_gpu = GPUInitializer()


def _set_gpu_impl(gpu_id: Union[int, List[int]]) -> None:
    """使用するGPUの設定
    """
    if isinstance(gpu_id, int):
        gpu_id = [gpu_id]

    for id in gpu_id:
        clear_gpu(id)
    if tf.__version__ >= "2.1.0":
        physical_devices = tf.config.list_physical_devices('GPU')
        devices = [physical_devices[id] for id in gpu_id]
        tf.config.set_visible_devices(devices, 'GPU')
        for device in devices:
            tf.config.experimental.set_memory_growth(device, True)
    elif tf.__version__ >= "2.0.0":
        #TF2.0
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        devices = [physical_devices[id] for id in gpu_id]
        tf.config.experimental.set_visible_devices(devices, 'GPU')
        for device in devices:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        from keras.backend.tensorflow_backend import set_session  # type: ignore
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(  # type: ignore
            visible_device_list=[str(id) for id in gpu_id], allow_growth=True))
        set_session(tf.Session(config=config))  # type: ignore
    clef.logger.info("Set visible gpu : {}".format(gpu_id))
