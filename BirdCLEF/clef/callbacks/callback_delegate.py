from typing import Optional, Dict, Any


class CallbackDelegate(object):

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_epoch_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_step_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_step_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_validation_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_validation_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_epoch_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass
