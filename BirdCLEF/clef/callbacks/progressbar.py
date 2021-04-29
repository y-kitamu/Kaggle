from typing import Dict, Optional, Any

from clef.callbacks import CallbackDelegate


class ProgressBar(CallbackDelegate):
    """step毎の学習の進捗状況を表示
    """

    def on_step_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        stats = ", ".join([
            "{} = {:.3f}".format(metric.name,
                                 metric.result().numpy()) for metric in self.trainer.train_metrics
        ])
        print("\r {} / {} ({})".format(logs[self.trainer.step], self.trainer.steps_per_epoch, stats),
              end="")

    def on_step_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        print("\r", end="")
