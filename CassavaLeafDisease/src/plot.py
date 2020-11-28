import math

import numpy as np
import matplotlib.pyplot as plt

# def show_random_selected_images(random_state=None):


def plot_batch_images(images, labels=None, cols=10):
    rows = math.ceil(images.shape[0] / 10)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 5, rows * 5))

    if images.dtype == np.float64 or images.dtype == np.float32:
        images = (images * 255).astype(np.uint8)

    for i in range(images.shape[0]):
        row = int(i / cols)
        col = i % cols
        axes[row][col].imshow(images[i][:, :, ::-1])
        if labels is not None:
            axes[row][col].set_title("Trur label = {}".format(labels[i]))
    fig.show()
    return
