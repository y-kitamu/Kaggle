"""test_arcface.py
"""
import numpy as np
import pytest
import tensorflow as tf
import ykaggle_core as ycore


def test_arcface():
    # initialize
    arcface = ycore.models.components.ArcFace(3, 0.1)
    x = tf.ones([3, 2], dtype=tf.float32)
    arcface(x)

    # run test
    # pred norm_x = [[0.6, 0.8], [0, 1], [1 / 1.4, 1 / 1.4]]
    x = tf.constant(
        [
            [3, 4],
            [0, 1],
            [2, 2],
        ],
        dtype=tf.float32,
    )
    # pred norm w = [[0.6, 0, 1], [0.8, 1, 0]]
    arcface.w = tf.constant(
        [
            [3, 0, 1],
            [4, 1, 0],
        ],
        dtype=tf.float32,
    )
    # pred output [[1.0, 0.8, 0.6], [0.8, 1, 0], [1.0, 1/1.4, 1/1.4]]
    res = arcface(x)

    pred = np.array([[1.0, 0.8, 0.6], [0.8, 1.0, 0.0], [0.9899495, 0.70710677, 0.70710677]])
    assert ((res.numpy() - pred) < 1e-5).all()
