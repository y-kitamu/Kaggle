from typing import Optional

import tensorflow as tf

import clef
from clef.trainer import Trainer
from clef.evaluator import Evaluator
from clef import config_definitions


class Controller(object):
    """Class that controls the outer loop of model training and evaluation.
    """

    def __init__(self,
                 config: config_definitions.ControllerConfig,
                 trainer: Trainer = None,
                 evaluator: Evaluator = None) -> None:
        self._config = config
        self._trainer = trainer
        self._evaluator = evaluator
        self.strategy = self.get_strategy()

    def get_strategy(self) -> tf.distribute.Strategy:
        if self.config.strategy == "mirrored":
            return tf.distribute.MirroredStrategy()
        return tf.distribute.get_strategy()

    def train(self, epochs: int) -> None:
        if self.trainer is None:
            clef.logger.warning("Trainer object is not set in Controller. Abort `train`")
            return

        self.trainer.compile(self.strategy)

        self.trainer.on_train_begin()
        for i in range(epochs):
            self.trainer.train(i, self.strategy)
            self.trainer.validation(i, self.strategy)
        self.trainer.on_train_end()

    def evaluate(self) -> None:
        pass

    # TODO : remove boilerplate
    @property
    def config(self) -> config_definitions.ControllerConfig:
        return self._config

    @property
    def trainer(self) -> Optional[Trainer]:
        return self._trainer

    @property
    def evaluator(self) -> Optional[Evaluator]:
        return self._evaluator
