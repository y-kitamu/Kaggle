"""multiprocess generator
"""
import queue
import multiprocessing as mp
from typing import Any, List, Optional, Tuple

import numpy as np
import cv2

import clef


def read_and_process_image(file_list: List[str],
                           labels: Optional[List[Any]] = None,
                           output_shape: Tuple[int, int] = None,
                           process_func=None):
    """
    Args:
        file_list (list of str)  :
        labels (list of int) :
        output_shape (tuple of int) : output image shape (H, W).
        process_func (callable) : preprocess function (input : image, output : image)
    """
    img_list = []
    for fname in file_list:
        img = cv2.imread(fname)
        if img is None:
            clef.logger.warning("File does not exist : {}".format(fname))
            continue
        img = process_func(img)
        img = cv2.resize(img, output_shape)
        img_list.append(img)
    return img_list, labels


class MPGenerator:
    """Read and process images on multi process.
    Get value
    """

    def __init__(self,
                 file_list: List[str],
                 batch_size: int,
                 labels: Optional[List[Any]] = None,
                 num_process=8,
                 preprocess_func=None,
                 num_prefetch=16):
        self.file_list = np.array(file_list)
        self.batch_size = batch_size
        self.labels = labels
        self.num_process = num_process
        self.preprocess_func = preprocess_func
        self.num_prefetch = num_prefetch

    @property
    def steps_per_epoch(self):
        return int(len(self.file_list) / self.batch_size)

    def __len__(self):
        return len(self.file_list)

    def __iter__(self):
        que = queue.Queue(self.num_prefetch)
        pool = mp.Pool(self.num_process)
        try:
            while True:
                while que.not_full:
                    que.put(pool.apply_async(read_and_process_image, args=()))
                yield que.get().get()
        except:
            clef.logger.warning("Finish generator by exception")
        pool.terminate()
