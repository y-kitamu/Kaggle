import time

from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from tensorflow.python.keras import backend as K


class ProgressLogger(Callback):

    def __init__(self, metrics=["loss", "accuracy"]):
        super(ProgressLogger, self).__init__()
        self.metrics = metrics

    def on_train_begin(self, logs):
        self.epochs = logs["epochs"]
        self.steps_per_epochs = logs["steps_per_epochs"]
        self.num_digits = len(str(self.steps_per_epochs))

    def on_train_batch_end(self, step, logs=None):
        output = ""
        for metrix in self.metrics:
            output += " - {} : {:.4f}".format(metrix, logs[metrix])
        print(("\r{:>" + str(self.num_digits) + "} / {} {}").format(step, self.steps_per_epochs, output),
              end="")

    def on_epoch_begin(self, epoch, logs=None):
        print("Epoch {} / {}".format(epoch, self.epochs))
        self.start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        output = ""
        for metrix in self.metrics:
            output += ", {} : {:.4f}".format(metrix, logs[metrix])
        for metrix in self.metrics:
            output += ", {} : {:.4f}".format("val_" + metrix, logs["val_" + metrix])

        elapsed = time.time() - self.start
        print("\r  {}, Elapsed time : {:.3f} sec".format(output, elapsed))


class LRScheduler(ReduceLROnPlateau):

    def __init__(self,
                 monitor='val_loss',
                 factor=0.33,
                 patience=2,
                 verbose=1,
                 mode='auto',
                 min_delta=1e-4,
                 cooldown=0,
                 min_lr=1e-6,
                 warmup=1,
                 **kwargs):
        super().__init__(monitor=monitor,
                         factor=factor,
                         patience=patience,
                         verbose=verbose,
                         mode=mode,
                         min_delta=min_delta,
                         cooldown=cooldown,
                         min_lr=min_lr,
                         **kwargs)
        self.warmup = warmup

    def set_model(self, *args, **kwargs):
        super().set_model(*args, **kwargs)
        self.default_lr = float(K.get_value(self.model.optimizer.lr))

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if epoch < self.warmup:
            print("\n warmup lr : {}".format(self.min_lr))
            K.set_value(self.model.optimizer.lr, self.min_lr)
            logs["lr"] = self.min_lr
        if epoch == self.warmup:
            print("\n finish warmup : {}".format(self.default_lr))
            K.set_value(self.model.optimizer.lr, self.default_lr)
            logs["lr"] = self.default_lr
