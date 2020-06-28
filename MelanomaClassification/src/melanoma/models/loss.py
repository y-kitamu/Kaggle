import chainer
import chainer.functions as F
from chainer import reporter
import numpy as np


class BaseLoss():

    def __init__(self, model):
        self.model = model

    def __call__(self, *args, **kwargs):
        t = args[-1]
        x = self.model.forward(args[0])

        loss = self._loss_func(x, t)
        reporter.report({'loss': loss}, self.model)

        with chainer.cuda.get_device_from_array(t):
            accuracy = F.accuracy(x, t.argmax(axis=1))
        reporter.report({'accuracy': accuracy}, self.model)
        return loss


class SigmoidLoss(BaseLoss):

    def _loss_func(self, x, t):
        loss = F.sigmoid_cross_entropy(x, t)
        return loss


class MixupLoss(BaseLoss):

    def __call__(self, *args, **kwargs):
        if chainer.config.train:
            lam = np.random.beta(self.model.mixup_alpha, self.model.mixup_alpha)
            x, t_a, t_b = self.model.forward(args[0], args[1], lam)
            loss = self._loss_func(x, t_a, t_b, lam)
            with chainer.cuda.get_device_from_array(t_a):
                accuracy = F.accuracy(x, (lam * t_a + (1 - lam) * t_b).argmax(axis=1))
        else:
            x = self.model.forward(args[0])
            t = args[1]
            loss = F.sigmoid_cross_entropy(x, t)
            with chainer.cuda.get_device_from_array(t):
                accuracy = F.accuracy(x, t.argmax(axis=1))

        reporter.report({'loss': loss}, self.model)
        reporter.report({'accuracy': accuracy}, self.model)
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
        ce_loss = F.softmax_cross_entropy(x, t.argmax(axis=1), reduce='no')
        pt = F.exp(-ce_loss)
        loss = F.mean((1 - pt)**self.gamma * ce_loss)
        return loss
