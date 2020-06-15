import chainer
from chainer.backends import cuda
import numpy as np
import cv2


class Predictor(chainer.Chain):

    def __init__(self, extractor, img_size, preprocess=None):
        """Wrapper that adds a prediction method to a feature extraction links
        Args:
            extractor (chainer.Link) : feature extraction model
            img_size (tuple) : tuple of image size (width, height) that input to extractor
            preprocess (callabel) : image preprocess function
        """
        super().__init__()
        self.img_size = img_size
        self.preprocess = preprocess
        with self.init_scope():
            self.extractor = extractor

    def _prepare(self, img):
        # [height, width, channel] -> [channel, height, width]
        if img.shape[0] == 3 or img.shape[0] == 1:
            img = img.transpose(1, 2, 0).astype(np.uint8)
        img = cv2.resize(img, self.img_size)
        img = img.transpose(2, 0, 1).astype(np.float32)
        if self.preprocess is not None:
            img = self.preprocess(img)
        return img

    def predict(self, imgs, **kwargs):
        """
        imgs (np.ndarray) : input image(s) (opencv BGR image, channel last : [height, width, channle])
                            dimension must be 3 (H, W, C) or 4 (B, H, W, C)
        """
        if len(imgs.shape) == 3:
            imgs = imgs[np.newaxis, ...]
        imgs = cuda.to_cpu(imgs)
        imgs = self.xp.asarray([self._prepare(img) for img in imgs])

        features = self.extractor.forward(imgs, **kwargs)

        if isinstance(features, tuple):
            output = []
            for feature in features:
                feature = feature.array
                output.append(cuda.to_cpu(feature.array))
            output = tuple(output)
        else:
            output = cuda.to_cpu(features.array)

        return output
