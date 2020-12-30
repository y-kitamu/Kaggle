import os
import math

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.constant import TRAIN_CSV, TRAIN_DATA_DIR


def plot_batch_images(images, labels=None, cols=10):
    """Plot images.
    Args:
        images (np.ndarray) : 4D array of images. [Batch, Height, Width, Channel]
        labels (np.ndarray) : True labels
        cols (int)          : number of images to be plotted in one line
    """
    rows = max(2, math.ceil(images.shape[0] / cols))
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 5, rows * 5))

    if images.dtype == np.float64 or images.dtype == np.float32:
        images = (images * 255).astype(np.uint8)

    for i in range(images.shape[0]):
        row = int(i / cols)
        col = i % cols
        axes[row][col].imshow(images[i][:, :, ::-1])
        if labels is not None:
            axes[row][col].set_title("Trur label = {}".format(labels[i]))
        axes[row][col].set_xticks([])
        axes[row][col].set_yticks([])
    fig.show()
    return


def plot_images_per_class(csv_filename=TRAIN_CSV, image_dir=TRAIN_DATA_DIR, num_max_plot=30):
    """Read csv file and plot each images per each class
    Args:
        csv_filename (str) :
        image_dir (str)    :
    """
    df = pd.read_csv(csv_filename)

    for label in sorted(df.label.unique()):
        fnames = df[df.label == label].image_id.to_numpy()
        images = np.stack([
            cv2.imread(os.path.join(image_dir, bname))
            for bname in np.random.choice(fnames, num_max_plot, replace=False)
        ])
        labels = np.ones(num_max_plot) * label
        plot_batch_images(images, labels)
