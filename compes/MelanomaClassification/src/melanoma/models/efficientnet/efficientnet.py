import math
from collections import namedtuple

import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np

from melanoma.models.efficientnet.mbconv import MBConvBlock
from melanoma.functions import mixup

GlobalParams = namedtuple("GlobalParams", [
    "width_coefficient",
    "depth_coefficient",
    "depth_divisor",
    "min_depth",
    "batch_norm_epsilon",
    "dropout_rate",
    "drop_connect_rate",
])

BlockArgs = namedtuple("BlockArgs", [
    "kernel_size",
    "num_repeat",
    "input_filters",
    "output_filters",
    "expand_ratio",
    "id_skip",
    "stride",
    "se_ratio",
])

EfficientNetB0 = GlobalParams(
    width_coefficient=1.0,
    depth_coefficient=1.0,
    depth_divisor=8,
    min_depth=None,
    batch_norm_epsilon=1e-3,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
)

EfficientNetB3 = GlobalParams(
    width_coefficient=1.2,
    depth_coefficient=1.4,
    depth_divisor=8,
    min_depth=None,
    batch_norm_epsilon=1e-3,
    dropout_rate=0.3,
    drop_connect_rate=0.2,
)

DEFAULT_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3,
              num_repeat=1,
              input_filters=32,
              output_filters=16,
              expand_ratio=1,
              id_skip=True,
              stride=1,
              se_ratio=0.25),
    BlockArgs(kernel_size=3,
              num_repeat=2,
              input_filters=16,
              output_filters=24,
              expand_ratio=6,
              id_skip=True,
              stride=2,
              se_ratio=0.25),
    BlockArgs(kernel_size=5,
              num_repeat=2,
              input_filters=24,
              output_filters=40,
              expand_ratio=6,
              id_skip=True,
              stride=2,
              se_ratio=0.25),
    BlockArgs(kernel_size=3,
              num_repeat=3,
              input_filters=40,
              output_filters=80,
              expand_ratio=6,
              id_skip=True,
              stride=2,
              se_ratio=0.25),
    BlockArgs(kernel_size=5,
              num_repeat=3,
              input_filters=80,
              output_filters=112,
              expand_ratio=6,
              id_skip=True,
              stride=1,
              se_ratio=0.25),
    BlockArgs(kernel_size=5,
              num_repeat=4,
              input_filters=112,
              output_filters=192,
              expand_ratio=6,
              id_skip=True,
              stride=2,
              se_ratio=0.25),
    BlockArgs(kernel_size=3,
              num_repeat=1,
              input_filters=192,
              output_filters=320,
              expand_ratio=6,
              id_skip=True,
              stride=1,
              se_ratio=0.25)
]


def round_filters(filters, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class EfficientNet(chainer.Chain):

    def __init__(self, num_classes=2, blocks_args=DEFAULT_BLOCKS_ARGS, global_params=EfficientNetB0):
        super().__init__()
        assert isinstance(blocks_args, list)
        assert len(blocks_args) > 0
        self._global_params = global_params
        self.loss = None
        self.accuracy = None

        bn_eps = global_params.batch_norm_epsilon
        out_channels = round_filters(32, global_params)

        with self.init_scope():
            self._conv_stem = L.Convolution2D(None, out_channels, ksize=3, stride=2, nobias=True)
            self._bn0 = L.BatchNormalization(size=out_channels, eps=bn_eps)

            self._blocks = chainer.ChainList()
            for bargs in blocks_args:
                bargs = bargs._replace(input_filters=round_filters(bargs.input_filters, global_params),
                                       output_filters=round_filters(bargs.output_filters, global_params),
                                       num_repeat=round_repeats(bargs.num_repeat, global_params))
                self._blocks.add_link(MBConvBlock(bargs, global_params))
                if bargs.num_repeat > 1:
                    bargs = bargs._replace(input_filters=bargs.output_filters, stride=1)
                for _ in range(bargs.num_repeat - 1):
                    self._blocks.add_link(MBConvBlock(bargs, global_params))

            out_channels = round_filters(1280, global_params)
            self._conv_head = L.Convolution2D(bargs.output_filters, out_channels, ksize=1, nobias=True)
            self._bn1 = L.BatchNormalization(out_channels, eps=bn_eps)

            self._avg_pooling = lambda x: F.average(x, axis=(2, 3))
            self._dropout = lambda x: F.dropout(x, global_params.dropout_rate)
            self._fc = L.Linear(out_channels, num_classes)
            self._swish = F.sigmoid

    def forward(self, inputs, *args, **kwargs):
        """EfficientNet's forward function
        """
        x = self.extract_features(inputs)
        x = self._avg_pooling(x)
        x = self._dropout(x)
        x = self._fc(x)

        return x

    def extract_features(self, inputs):
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        x = self._swish(self._bn1(self._conv_head(x)))
        return x


class EfficientNetCW(EfficientNet):
    """EfficientNet : loss func with class weights version
    """

    def __init__(self,
                 num_classes=2,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 global_params=EfficientNetB0,
                 class_weights=None):
        super().__init__(num_classes, blocks_args, global_params)
        self.n_classes = num_classes
        self.class_weights = class_weights
        if self.class_weights is None:
            self.class_weights = np.array([1.0 / self.n_classes] * self.n_classes)


class EfficientNetMixUp(EfficientNet):
    """Mixup EfficientNet


    """

    def __init__(self,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 global_params=EfficientNetB0,
                 mixup_alpha=0.2,
                 mixup_layer_idx=None):
        super().__init__(blocks_args, global_params)
        self.mixup_alpha = mixup_alpha
        self.mixup_layer = mixup_layer_idx or np.random.randint(0, len(self._blocks))

    def forward(self, inputs, target=None, lam=None, **kwargs):
        """EfficientNet's forward function
        """
        is_train = chainer.config.train
        if is_train and self.mixup_layer == 0:
            x, t_a, t_b = mixup(inputs, target, lam)

        x = self._swish(self._bn0(self._conv_stem(inputs)))

        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if is_train and self.mixup_layer == idx:
                x, t_a, t_b = mixup(x, target, lam)

        x = self._swish(self._bn1(self._conv_head(x)))

        x = self._avg_pooling(x)
        x = self._dropout(x)
        x = self._fc(x)

        if is_train:
            return x, t_a, t_b
        return x
