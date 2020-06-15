import math
from collections import namedtuple

import chainer
import chainer.links as L
import chainer.functions as F
from chainer.functions import accuracy
from chainer import reporter
import numpy as np

GlobalParams = namedtuple("GlobalParams", [
    "width_coefficient",
    "depth_coefficient",
    "depth_divisor",
    "min_depth",
    "batch_norm_epsilon",
    "dropout_rate",
    "drop_connect_rate",
    "num_classes",
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
    num_classes=2,
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


class MBConvBlock(chainer.Chain):

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip

        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        bn_eps = global_params.batch_norm_epsilon

        self.swish = F.sigmoid

        with self.init_scope():
            if self._block_args.expand_ratio != 1:
                self._expand_conv = L.Convolution2D(in_channels=inp, out_channels=oup, ksize=1, nobias=True)
                self._bn0 = L.BatchNormalization(oup, eps=bn_eps)

            k = self._block_args.kernel_size
            s = self._block_args.stride
            self._depthwise_conv = L.Convolution2D(in_channels=oup,
                                                   out_channels=oup,
                                                   groups=oup,
                                                   ksize=k,
                                                   pad=k // 2,
                                                   stride=s,
                                                   nobias=True)
            self._bn1 = L.BatchNormalization(oup, eps=bn_eps)

            if self.has_se:
                n_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
                self._se_reduce = L.Linear(in_size=oup, out_size=n_squeezed_channels)
                self._se_expand = L.Linear(in_size=n_squeezed_channels, out_size=oup)

            final_oup = self._block_args.output_filters
            self._project_conv = L.Convolution2D(in_channels=oup, out_channels=final_oup, ksize=1, nobias=True)
            self._bn2 = L.BatchNormalization(final_oup, eps=bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        x = self._expand_layer(inputs)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self.swish(x)

        x = self._se_layer(x)

        x = self._project_conv(x)
        x = self._bn2(x)

        x = self._skip_and_drop_connect(x, inputs, drop_connect_rate)
        return x

    def _expand_layer(self, x):
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(x)
            x = self._bn0(x)
            x = self.swish(x)
        return x

    def _se_layer(self, x):
        if self.has_se:
            x_squeezed = F.average(x, axis=(2, 3))
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self.swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = F.sigmoid(x_squeezed)[..., None, None] * x
        return x

    def _skip_and_drop_connect(self, x, inputs, drop_connect_rate):
        input_filters = self._block_args.input_filters
        output_filters = self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = self._drop_connect(x, drop_connect_rate)
            x = x + inputs
        return x

    def _drop_connect(self, x, p, is_train=chainer.config.train):
        """Dropout??
        """
        if not is_train:
            return x
        keep_prob = 1 - p
        binary = self.xp.asarray(np.floor(np.random.rand(*x.shape) + keep_prob))
        output = x / keep_prob * binary
        return output


class EfficientNet(chainer.Chain):

    def __init__(self, blocks_args=DEFAULT_BLOCKS_ARGS, global_params=EfficientNetB0):
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
            self._fc = L.Linear(out_channels, global_params.num_classes)
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

    def loss_func(self, *args, **kwargs):
        t = args[-1]
        self.y = self.forward(*args, **kwargs)
        self.loss = F.sigmoid_cross_entropy(self.y, t)

        with chainer.cuda.get_device_from_array(t):
            self.accuracy = accuracy(self.y, t.argmax(axis=1))
        reporter.report({'loss': self.loss}, self)
        reporter.report({'accuracy': self.accuracy}, self)
        return self.loss


class EfficientNetCW(EfficientNet):
    """EfficientNet : loss func with class weights version
    """

    def __init__(self, blocks_args=DEFAULT_BLOCKS_ARGS, global_params=EfficientNetB0, class_weights=None):
        super().__init__(blocks_args, global_params)
        self.n_classes = global_params.num_classes
        if class_weights:
            self.class_weights = class_weights
        else:
            np.array([1.0 / self.n_classes] * self.n_classes)

    def loss_func(self, *args, **kwargs):
        t = args[-1]
        self.y = self.forward(*args, **kwargs)
        losses = F.sigmoid_cross_entropy(self.y, t, reduce='no')
        labels = t.argmax(axis=1)
        self.loss = sum([(self.class_weights[i] * losses[labels == i]).sum() for i in self.n_classes])

        with chainer.cuda.get_device_from_array(t):
            self.accuracy = accuracy(self.y, t.argmax(axis=1))
        reporter.report({'loss': self.loss}, self)
        reporter.report({'accuracy': self.accuracy}, self)
        return self.loss
