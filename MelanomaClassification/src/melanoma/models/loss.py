import chainer
import chainer.functions as F
from chainer import reporter


class SigmoidLoss():

    def __init__(self, model):
        self.model = model

    def __call__(self, *args, **kwargs):
        t = args[-1]
        x = self.model.forward(args[0])
        return self._loss_func(x, t)

    def _loss_func(self, x, t):
        loss = F.sigmoid_cross_entropy(x, t)
        with chainer.cuda.get_device_from_array(t):
            accuracy = F.accuracy(x, t.argmax(axis=1))

        reporter.report({'loss': loss}, self.model)
        reporter.report({'accuracy': accuracy}, self.model)
        return loss
