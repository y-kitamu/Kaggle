from typing import List, Optional, Any, Dict, TYPE_CHECKING

from clef.callbacks import CallbackDelegate
if TYPE_CHECKING:
    from clef.trainer import Trainer


class CallbackList(CallbackDelegate):

    def __init__(self, callback_list: List[CallbackDelegate]):
        self.callback_list = callback_list

    def set_trainer(self, trainer: "Trainer"):
        for callback in self.callback_list:
            callback.set_trainer(trainer)

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in self.callback_list:
            callback.on_train_begin(logs)

    def on_epoch_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in self.callback_list:
            callback.on_epoch_begin(logs)

    def on_step_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in self.callback_list:
            callback.on_step_begin(logs)

    def on_step_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in self.callback_list:
            callback.on_step_end(logs)

    def on_validation_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in self.callback_list:
            callback.on_validation_begin(logs)

    def on_validation_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in self.callback_list:
            callback.on_validation_end(logs)

    def on_epoch_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in self.callback_list:
            callback.on_epoch_end(logs)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in self.callback_list:
            callback.on_train_end(logs)
