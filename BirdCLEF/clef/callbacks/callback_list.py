from typing import List

from clef.callbacks.callback_delegate import CallbackDelegate


class CallbackList(CallbackDelegate):

    def __init__(callback_list: List[CallbackDelegate]):
        self.callback_list = callback_list

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in callback_list:
            callback.on_train_begin(logs)

    def on_epoch_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in callback_list:
            callback.on_epoch_begin(logs)

    def on_step_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in callback_list:
            callback.on_step_begin(logs)

    def on_step_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in callback_list:
            callback.on_step_end(logs)

    def on_validation_begin(self, logs: Optional[Dcit[str, Any]] = None) -> None:
        for callback in callback_list:
            callback.on_validation_begin(logs)

    def on_validation_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in callback_list:
            callback.on_validation_end(logs)

    def on_epoch_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in callback_list:
            callback.on_epoch_end(logs)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in callback_list:
            callback.on_train_end(logs)
