import os
from typing import Optional, Dict, Any, TYPE_CHECKING

import clef
from clef.callbacks.callback_delegate import CallbackDelegate
if TYPE_CHECKING:
    from clef.trainer import Trainer


class ModelCheckPoint(CallbackDelegate):

    def __init__(self, output_dir, metric_name: str, maximize: bool = True, prefix=None) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metric_name = metric_name
        self.maximize = maximize
        self.best = None
        self.prefix = prefix
        self.metric = None

    def set_trainer(self, trainer: "Trainer") -> None:
        super().set_trainer(trainer)
        metrics = self.trainer.train_metrics + self.trainer.validation_metrics
        for metric in metrics:
            if metric.name == self.metric_name:
                self.metric = metric
                return
        clef.logger.info("Metric of name == `{}` not found in trainer".format(self.metric_name))

    def on_epoch_end(self, logs: Optional[Dict[str, Any]]) -> None:
        if self.prefix is None:
            basename = "{}.h5".format(self.metric_name)
        else:
            basename = "{}_{}.h5".format(self.prefix, self.metric_name)
        output_fname = os.path.join(self.output_dir, basename)

        is_save = self.metric is None
        if self.metric:
            val = self.metric.result().numpy()
            if self.best is None:
                self.best = self.metric.result().numpy()
                is_save = True
            else:
                diff = val - self.best if self.maximize else self.best - val
                if diff > 0:
                    self.best = val
                    is_save = True
        if is_save:
            clef.logger.info("Save weights : {}".format(output_fname))
            self.trainer.model.save_weights(output_fname)
