from official.core.base_task import Task

from clef.model import create_simple_model


class MnistTask(Task):
    def build_model(self):
        create_simple_model()
