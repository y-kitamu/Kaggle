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

    def get_strategy(self):
        if self.config.strategy == "mirrored":
            return tf.distribute.MirroredStrategy()

    def train(self, epochs: int) -> None:
        if self.trainer is None:
            clef.logger.warning("Trainer object is not set in Controller. Abort `train`")
            return

        self.trainer.compile(self.strategy)

        self.trainer.on_train_begin()
        for i in range(epochs):
            self.trainer.train()
        self.trainer.on_train_end()

    def evaluate(self) -> None:
        pass

    # TODO : remove boilerplate
    @property
    def config(self):
        return self._config

    @property
    def trainer(self):
        return self._trainer

    @property
    def evaluator(self):
        return self._evaluator
