import time

from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau


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
                 patience=3,
                 verbose=1,
                 mode='auto',
                 min_delta=1e-4,
                 cooldown=0,
                 min_lr=1e-6,
                 warmup=1,
                 **kwargs):
        super().__init__(monitor, factor, patience, verbose, mode, min_delta, cooldown, min_lr, **kwargs)
        self.warmup = warmup

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.warmup:
            logs = logs or {}
            logs["lr"] = self.min_lr
            self.model.optimizer.lr = self.min_lr
        else:
            super().on_epoch_end(epoch, logs)
