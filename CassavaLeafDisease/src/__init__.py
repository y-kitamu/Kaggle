import os, glob

VERSION = (1, 0, 0)

import tensorflow as tf

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

from src import constant
from src import dataset
from src import model
from src import train
from src import utility
from src import predict
from src import visualize

__all__ = ["constant", "dataset", "model", "train", "utility", "predict", "visualize"]
