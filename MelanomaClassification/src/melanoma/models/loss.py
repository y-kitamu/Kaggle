import chainer
import chainer.functions as F
from chainer import reporter


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
