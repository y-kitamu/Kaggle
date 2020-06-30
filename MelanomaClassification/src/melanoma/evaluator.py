from chainer.training.extensions import Evaluator
from sklearn.metrics import roc_auc_score
import cupy as cp


class CustomEvaluator(Evaluator):
    """Default evaluator + ROC-AUC score
    """

    def __call__(self, trainer=None):
        result = super(CustomEvaluator, self).__call__(trainer)
        if hasattr(self.eval_func, "true_labels") and hasattr(self.eval_func, "confs"):
            roc = roc_auc_score(cp.asnumpy(self.eval_func.true_labels), cp.asnumpy(self.eval_func.confs[0]))
            print(f"ROC : {roc:.3f}")
        return result
