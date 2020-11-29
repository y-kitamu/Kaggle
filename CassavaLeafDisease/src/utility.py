import tensorflow as tf
import pdb, traceback, sys


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


def set_gpu(gpu_id=0):
    if gpu_id < 0:
        return
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
