"""__init__.py
"""
import glob
import logging
import os
import sys
from pathlib import Path
from typing import List, Union

import tensorflow as tf

# constants
PROJECT_ROOT = Path(__file__).parents[1]
TENSORBOARD_LOG_DIR = PROJECT_ROOT / "logs"

## Logger settings
logging.getLogger().setLevel(logging.DEBUG)

LOGGER_NAME = __name__
DEFAULT_LOGLEVEL = logging.DEBUG
SHORT_FORMATTER = logging.Formatter(
    fmt="[%(asctime)s %(levelname)s %(pathname)s at line %(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LONG_FORMATTER = logging.Formatter(
    fmt="[%(asctime)s %(levelname)s %(pathname)s in %(funcName)s at line %(lineno)d] %(message)s"
)
logger = logging.getLogger(LOGGER_NAME)


def enable_logging_to_stdout(log_level=DEFAULT_LOGLEVEL, formatter=SHORT_FORMATTER):
    """logging to stdout
    Args:
        log_level (int) : log level
        formatter (logging.Formatter) : log format
    """
    # remove old handler
    for handler in list(logger.handlers):
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)

    # add new handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("Set new `StreamHandler` instance to the logger.")


def enable_logging_to_file(
    filename, remove_old_handler=True, log_level=DEFAULT_LOGLEVEL, formatter=LONG_FORMATTER
):
    """logging to file
    Args:
        filename (str) :
        remove_old_handler (bool) : If True、remove old instance of `logging.FileHandler` from `logger`
        log_level (int) :
        format (logging.Formatter) :
    """
    # remove old handler
    if remove_old_handler:
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)

    # add new handler
    dirname = os.path.abspath(os.path.dirname(filename))
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        logger.warning("Directory does not exist. Create directory = {}".format(dirname))
    fh = logging.FileHandler(filename, encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info(
        "Set new `FileHandler` instance to the logger. output file = {}".format(
            os.path.abspath(filename)
        )
    )


def remove_logfile(logdir, max_save=10):
    logs = sorted(glob.glob(os.path.join(logdir, "*.log")))
    if len(logs) > max_save:
        for logfile in logs[:-max_save]:
            try:
                os.remove(logfile)
            except Exception:
                continue


enable_logging_to_stdout(log_level=logging.DEBUG)

# GPU settings
def set_gpu(gpu_id: Union[int, List[int]]) -> None:
    """Set gpus to use."""
    if isinstance(gpu_id, int):
        gpu_id = [gpu_id]

    physical_devices = tf.config.list_physical_devices("GPU")
    devices = [physical_devices[id] for id in gpu_id]
    tf.config.set_visible_devices(devices, "GPU")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)


# improt modules
from . import config, losses, models, optimizers
