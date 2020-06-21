import six
import chainer
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from melanoma.extensions import LRScheduler


def serialize(file_path, obj):
    print("save {}".format(file_path))
    chainer.serializers.save_npz(file_path, obj)


class TrainerBuilder():
    """
    """

    def __init__(self, updater, epoch, evaluator, output_dir="results"):
        self.trainer = training.Trainer(updater, (epoch, 'epoch'), out=output_dir)
        self.evaluator = evaluator

    def build(self):
        self._set_extensions()
        return self.trainer

    def _set_extensions(self):
        target = self.trainer.updater.get_all_optimizers()["main"].target

        log_interval = (1, 'epoch')
        print_interval = (1, 'epoch')
        snapshot_interval = (10, 'epoch')

        self.trainer.extend(self.evaluator, trigger=log_interval, priority=training.extension.PRIORITY_WRITER)
        self.trainer.extend(chainer.training.extensions.observe_lr(), trigger=print_interval)
        for name, optimizer in six.iteritems(self.trainer.updater.get_all_optimizers()):
            if name != "main":
                continue
            self.trainer.extend(extensions.snapshot_object(optimizer.target, 'snapshot_model_{.updater.epoch}.npz'),
                                trigger=snapshot_interval)
            self.trainer.extend(
                extensions.snapshot_object(optimizer.target, 'snapshot_model_loss.npz', savefun=serialize),
                trigger=triggers.MinValueTrigger("validation/main/loss", trigger=log_interval),
                priority=training.extension.PRIORITY_READER,
            )
            self.trainer.extend(
                extensions.snapshot_object(optimizer.target, "snapshot_model_accuracy.npz", savefun=serialize),
                trigger=triggers.MaxValueTrigger("validation/main/accuracy", trigger=log_interval),
                priority=training.extension.PRIORITY_READER,
            )
        self.trainer.extend(extensions.ProgressBar())
        self.trainer.extend(extensions.LogReport(trigger=print_interval))
        self.trainer.extend(extensions.PrintReport([
            'iteration', 'epoch', 'elapsed_time', 'lr', 'main/loss', 'validation/main/loss', 'main/accuracy',
            'validation/main/accuracy'
        ]),
                            trigger=print_interval)

        self.trainer.extend(
            extensions.PlotReport(["main/loss", "validation/main/loss"], x_key="epoch", filename="loss.png"))
        self.trainer.extend(
            extensions.PlotReport(["main/accuracy", "validation/main/accuracy"],
                                  x_key="epoch",
                                  filename="accuracy.png"))
        # self.trainer.extend(LRScheduler.WarmUpCosineAnnealing("alpha"))
