import chainer
import chainer.functions as F
import chainer.links as L


class Net(chainer.Chain):

    def __init__(self, extractor, n_meta_features):
        super().__init__()
        with self.init_scope():
            self.extractor = extractor

    def forward(self, *args, **kwargs):
        x, meta = args[0], args[1]
        cnn_features = self.extractor(x)
        if isinstance(cnn_features, dict):
            return cnn_features["prob"]
        return cnn_features

    def to_gpu(self, device):
        super().to_gpu(device)
        self.extractor.to_gpu(device)

    def to_cpu(self):
        super().to_cpu()
        self.extractro.to_cpu()


class NetCM(chainer.Chain):

    def __init__(self, extractor, n_meta_features):
        super().__init__()
        with self.init_scope():
            self.extractor = extractor
            self.extractor._fc = L.Linear(None, 500)
            self.meta_fc0 = L.Linear(n_meta_features, 500)
            self.meta_bn0 = L.BatchNormalization(500)
            self.meta_fc1 = L.Linear(500, 250)
            self.meta_bn1 = L.BatchNormalization(250)
            self.output = L.Linear(500 + 250, 1)
            self.output = L.Linear(None, 1)

    def forward(self, *args, **kwargs):
        x, meta = args[0], args[1]
        cnn_features = self.extractor(x)
        if isinstance(cnn_features, dict):
            cnn_features = cnn_features["prob"]
        output = self.output(cnn_features)
        meta = F.relu(self.meta_bn0(self.meta_fc0(meta)))
        meta_features = F.relu(self.meta_bn1(self.meta_fc1(meta)))
        features = F.concat([cnn_features, meta_features])
        output = self.output(features)
        return output

    def to_gpu(self, device):
        super().to_gpu(device)
        self.extractor.to_gpu(device)

    def to_cpu(self):
        super().to_cpu()
        self.extractro.to_cpu()
