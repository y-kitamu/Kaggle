import numpy as np
import cv2

from melanoma import constants


def normalize(img):
    return (img - constants.IMAGE_MEAN) / constants.IMAGE_STD


def augmentations(img):
    augs = [standard_augmentation, add_noise]
    for aug in augs:
        img = aug(img)
    return img


def add_noise(img, scale=10):
    """Add gaussian noise to input image `img`
    Args:
       img (np.ndarray) : input image
    Return : np.ndarray
    """
    img += ((np.random.random(img.shape) * 2 - 1) * 10).astype(img.dtype)
    return img


def standard_augmentation(img):
    """
    """
    img = vertical_flip(img)
    img = horizontal_flip(img)
    # img = vertical_shift(img)
    # img = horizontal_shift(img)
    return img


def vertical_flip(img):
    """
    """
    if np.random.random() > 0.5:
        return img[::-1]
    return img


def horizontal_flip(img):
    """
    """
    if np.random.random() > 0.5:
        return img[:, ::-1]
    return img


def vertical_shift(img, shift_range=0.125):
    """
    Args:
        img (np.ndarray) :
        shift_range (float) : ratio of max vertical shift to image height.
    """
    if np.random.random() > 0.5:
        return img
    shift = int((np.random.random() * 2 - 1) * shift_range * img.shape[0])
    if shift < 0:
        img[:shift] = img[-shift:]
        img[shift:] = 0
    elif shift > 0:
        img[:-shift] = img[shift:]
        img[-shift:] = 0
    return img


def horizontal_shift(img, shift_range=0.125):
    """
    Args:
        img (np.ndarray) :
        shift_range (float) : ratio of max horizontal shift to image width.
    """
    if np.random.random() > 0.5:
        return img
    shift = int((np.random.random() * 2 - 1) * shift_range * img.shape[1])
    if shift < 0:
        img[:, :shift] = img[:, -shift:]
        img[:, shift:] = 0
    elif shift > 0:
        img[:, :-shift] = img[:, shift:]
        img[:, -shift:] = 0
    return img
