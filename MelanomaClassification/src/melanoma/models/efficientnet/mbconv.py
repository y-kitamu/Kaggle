import chainer
import chainer.links as L
import chainer.functions as F


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
        binary = self.xp.floor(self.xp.random.rand(*x.shape)) + keep_prob
        output = x / keep_prob * binary
        return output
