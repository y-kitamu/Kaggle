import chainer
import chainer.functions as F
from chainer import reporter
import numpy as np


class BaseLoss():

    def __init__(self, model):
        self.model = model
        self.true_labels = None
        self.confs = None

    def __call__(self, *args, **kwargs):
        x = self.model.forward(*args[:-1])

        loss = self._loss_func(x, args[-1])
        reporter.report({'loss': loss}, self.model)

        t = args[-1] if len(args[-1].shape) == 1 else args[-1].argmax(axis=-1)
        with chainer.cuda.get_device_from_array(t):
            if x.shape[1] == 1:
                accuracy = sum((x.data.flatten() > 0).astype(int) == t) / x.shape[0]
            else:
                accuracy = F.accuracy(x, t)
            reporter.report({'accuracy': accuracy}, self.model)

        if not chainer.config.train:
            if self.true_labels is None:
                self.true_labels = t
            else:
                self.true_labels = self.model.xp.hstack((self.true_labels, t))
            if self.confs is None:
                self.confs = F.sigmoid(x).data.transpose()
            else:
                self.confs = self.model.xp.hstack((self.confs, F.sigmoid(x).data.transpose()))
        return loss


class SigmoidLoss(BaseLoss):

    def _loss_func(self, x, t):
        if x.shape[1] == 1:
            x = x.data.flatten()
        loss = F.sigmoid_cross_entropy(x, t)
        return loss


class SoftmaxLoss(BaseLoss):

    def _loss_func(self, x, t):
        if len(t.shape) > 1:
            t = t.argmax(axis=-1)
        loss = F.softmax_cross_entropy(x, t)
        return loss


class MixupLoss(BaseLoss):

    def __call__(self, *args, **kwargs):
        if chainer.config.train:
            lam = np.random.beta(self.model.mixup_alpha, self.model.mixup_alpha)
            x, t_a, t_b = self.model.forward(args[0], args[1], lam)
            loss = self._loss_func(x, t_a, t_b, lam)
            with chainer.cuda.get_device_from_array(t_a):
                accuracy = F.accuracy(x, (lam * t_a + (1 - lam) * t_b).argmax(axis=1))
            reporter.report({'loss': loss}, self.model)
            reporter.report({'accuracy': accuracy}, self.model)
        else:
            super().__call__(args, kwargs)
        return loss

    def _loss_func(self, x, t_a, t_b, lam):
        loss = lam * F.sigmoid_cross_entropy(x, t_a) + (1 - lam) * F.sigmoid_cross_entropy(x, t_b)
        return loss


class FocalLoss(BaseLoss):
    """Focal Loss
    https://qiita.com/agatan/items/53fe8d21f2147b0ac982#fn1
    """

    def __init__(self, model, gamma=2):
        super().__init__(model)
        self.gamma = gamma

    def _loss_func(self, x, t):
        if len(t.shape) > 1:
            t = t.argmax(axis=-1)
        assert len(x.shape) == 2
        ce_loss = F.sigmoid_cross_entropy(x, t[..., None], reduce='no')
        pt = F.exp(-ce_loss)
        loss = F.mean((1 - pt)**self.gamma * ce_loss)
        return loss
