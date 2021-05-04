from typing import List, Tuple, Union

import tensorflow as tf
import numpy as np


def _bytes_features(value: Union[tf.Tensor, bytes]) -> tf.train.Feature:
    """byte配列をprotcol buffer形式のobjectに格納
    Args:
        value (tf.Tensor or bytes) :
    Return:
        (tf.train.Feature) :
    """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value: float) -> tf.train.Feature:
    """floatをprotcol buffer形式のobjectに格納
    Args:
        value (float) :
    Return:
        (tf.train.Feature) :
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value: int) -> tf.train.Feature:
    """intをprotcol buffer形式のobjectに格納
    Args:
        value (int) :
    Return:
        (tf.train.Feature) :
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_to_pb(image: np.ndarray, label: int) -> tf.train.Example:
    """画像(numpy array)をprotcol buffer形式のobjectに変換
    Args:
       image (np.ndarray) : 2d or 3d array ([H, W] or [H, W, C])
       label (int) :
    """
    depth = 1 if len(image.shape) == 2 else image.shape[2]
    feature = {
        "height": _int64_feature(image.shape[0]),
        "width": _int64_feature(image.shape[1]),
        "depth": _int64_feature(depth),
        "label": _int64_feature(label),
        "image_raw": _bytes_features(image.tobytes())
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_single_image(proto_object) -> Tuple[tf.Tensor, tf.Tensor]:
    """protcol buffer形式のデータを画像に変換
    Args:
        proto_object : A scalar string tensor, a single serialzied example
    Return:
        image (tf.Tensor) :
        label (tf.Tensor) :
    """
    features = tf.io.parse_single_example(proto_object,
                                          features={
                                              "height": tf.io.FixedLenFeature([], tf.int64),
                                              "width": tf.io.FixedLenFeature([], tf.int64),
                                              "depth": tf.io.FixedLenFeature([], tf.int64),
                                              "label": tf.io.FixedLenFeature([], tf.int64),
                                              "image_raw": tf.io.FixedLenFeature([], tf.string)
                                          })
    label = features["label"]
    image = tf.io.decode_raw(features["image_raw"], tf.uint8)
    image = tf.reshape(image, [features["height"], features["width"], features["depth"]])
    return image, label


def create_dataset_from_tfrecord(tfrecords: List[str]) -> tf.data.Dataset:
    """tfrecord形式のファイルからtf.data.Dataset objectを作成
    Args:
        tfrecords (list of str) : tfrecord形式のファイル名のリスト
    Return:
        dataset (tf.data.Dataset) :
    """
    dataset = tf.data.TFRecordDataset(tfrecords)
    return dataset.map(parse_single_image)


def write_images_to_tfrecord(images: np.ndarray, labels: np.ndarray, record_file: str) -> None:
    """画像をtfrecods形式のファイルに書き込み
    Args:
        images (np.ndarray) : 4D-array ([Batch, H, W, C]) or 3D-array ([Batch, H, W])
        labels (np.ndarray) : 1D-array ([Batch]).
        record_file (np.ndarray) : output filename
    Output:
        .tfrecords file : filename = `record_file`
    """
    with tf.io.TFRecordWriter(record_file) as writer:
        for image, label in zip(images, labels):
            tf_example = image_to_pb(image, label)
            writer.write(tf_example.SerializeToString())  # type: ignore
