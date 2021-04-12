import os
import pdb
import traceback
import sys
import multiprocessing

import tensorflow as tf
from numba import cuda


def run_debug(func):
    """Start pdb debugger at where the `func` throw Exception.
    """
    try:
        res = func()
        return res
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


def clear_gpu(gpu_id=0):
    cuda.select_device(gpu_id)
    device = cuda.get_current_device()
    device.reset()
    cuda.close()
    print("CUDA memory released: GPU {}".format(gpu_id))


def set_gpu(gpu_id=0):
    if gpu_id < 0:
        tf.config.set_visible_devices([], 'GPU')
        return
    clear_gpu(gpu_id)
    if tf.__version__ >= "2.1.0":
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)
    elif tf.__version__ >= "2.0.0":
        #TF2.0
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(physical_devices[gpu_id], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)
    else:
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(
            visible_device_list=str(gpu_id),  # specify GPU number
            allow_growth=True))
        set_session(tf.Session(config=config))


def run_as_multiprocess(func):

    def run(*args, **kwargs):
        # Pickle error occured when using mp.apply (or apply_async) instead of mp.Process.
        p = multiprocessing.Process(target=func, args=args, kwargs=kwargs)
        try:
            p.start()
            p.join()
            p.close()
        except:
            p.terminate()

    return run
