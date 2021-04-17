import time
from typing import Dict, Optional, Any

import clef
from clef.callbacks import CallbackDelegate


class Logger(CallbackDelegate):
    """Trainingの進行状況や各種metrics (accuracy, loss)、経過時間などをloggerに出力するクラス
    """

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        clef.logger.info("Start Training")

    def on_epoch_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.epoch_begin = time.time()
        clef.logger.info("".join(["=" for _ in range(40)]))
        clef.logger.info("Epoch {}".format(logs[self.trainer.epoch]))

    def on_epoch_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.epoch_end = time.time()
        header = "|".join(["{:7s}".format(""), "{:10s}".format("Loss")] +
                          ["{:10s}".format(key) for key in self.trainer.train_metrics.keys()])
        train_loss_str = "|".join([
            "{:7s}".format("Train"),
            "{:10s}".format("{:.3f}".format(self.trainer.train_loss.result().numpy()))
        ])
        train_metrics_str = "|".join([
            "{:10s}".format("{:.3f}".format(val.result().numpy()))
            for val in self.trainer.train_metrics.values()
        ])
        valid_loss_str = "|".join([
            "{:7s}".format("Valid"),
            "{:10s}".format("{:.3f}".format(self.trainer.validation_loss.result().numpy()))
        ])
        valid_metrics_str = "|".join([
            "{:10s}".format("{:.3f}".format(val.result().numpy()))
            for val in self.trainer.validation_metrics.values()
        ])
        clef.logger.info(header)
        clef.logger.info(train_loss_str + "|" + train_metrics_str)
        clef.logger.info(valid_loss_str + "|" + valid_metrics_str)
        clef.logger.info("Elapsed Time : {:.2f} second".format(self.epoch_end - self.epoch_begin))

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        clef.logger.info("End Training")
