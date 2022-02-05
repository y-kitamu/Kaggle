import chainercv
import numpy as np

from melanoma import constants


def standard_aug_transform(data):
    """training augmentations
    """
    img, metas, label = data

    # img_size = img.shape[1:]
    img = chainercv.transforms.random_flip(img, y_random=True, x_random=True)
    img = chainercv.transforms.pca_lighting(img, 25.5)

    # angle = np.random.randint(-10, 10)
    # img = chainercv.transforms.rotate(img, angle)

    # img = chainercv.transforms.random_expand(img, max_ratio=2)
    # img = chainercv.transforms.resize(img, img_size)

    return img, metas, label


def normalize_transform(data):
    img = data[0]
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    img = ((img - constants.IMAGE_MEAN) / constants.IMAGE_STD).transpose(2, 0, 1).astype(np.float32)
    return tuple([img] + [d for d in data[1:]])
