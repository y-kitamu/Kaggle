from chainer import reporter as reporter_module
from chainer import configuration
from chainer.training.extensions import Evaluator
from sklearn.metrics import roc_auc_score
import cupy as cp
import six


class CustomEvaluator(Evaluator):
    """Default evaluator + ROC-AUC score
    """

    def __call__(self, trainer=None):
        # set up a reporter
        reporter = reporter_module.Reporter()
        if self.name is not None:
            prefix = self.name + '/'
        else:
            prefix = ''
        for name, target in six.iteritems(self._targets):
            reporter.add_observer(prefix + name, target)
            reporter.add_observers(prefix + name, target.namedlinks(skipself=True))

        with reporter:
            with configuration.using_config('train', False):
                result = self.evaluate()

            if hasattr(self.eval_func, "true_labels") and hasattr(self.eval_func, "confs"):
                roc = roc_auc_score(cp.asnumpy(self.eval_func.true_labels), cp.asnumpy(self.eval_func.confs[-1]))
                self.eval_func.true_labels = None
                self.eval_func.confs = None
                result["validation/main/roc"] = roc
        reporter_module.report(result)
        return result
