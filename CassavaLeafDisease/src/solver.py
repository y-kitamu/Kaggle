import os
import math
import copy

import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils


class Solver(object):
    """ Run training and test
    Args:
        cfg (OmegaConf.DefaultDict)            : hydra parameter dictionary
        model (tf.keras.Model)                 : training (or test) model
        train_gen (tf.kefas.preprocessing.image.DataFrameIterator)
            : training iterator yielding (images, labels). images : [B, H, W, C], labels : [B, Num_Class]
        val_gen (tf.kefas.preprocessing.image.DataFrameIterator)  : validation generator
        loss_func (tf.keras.losses.Loss)       : loss function
        optimizer (tf.keras.optimizers)         : optimizer
        callbacks (list of tf.keras.callbacks.CallbackList) : callbacks
    """

    def __init__(self,
                 cfg,
                 model,
                 train_gen,
                 val_gen,
                 loss_func,
                 optimizer,
                 weights_path=None,
                 callbacks=tf.keras.callbacks.CallbackList()):
        self.cfg = cfg
        self.model = model
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.start_epoch = cfg["train"]["start_epoch"]
        self.callbacks = callbacks
        self.callbacks.set_model(self.model)

        self.load_weights(weights_path)
        self._prepare_metrix_containers()

    def _prepare_metrix_containers(self):
        self.loss = tf.keras.metrics.Mean(name='loss')
        self.accuracy = tf.keras.metrics.CategoricalAccuracy(name='accuracy')

    def _reset_state_of_metrix_containers(self):
        self.loss.reset_states()
        self.accuracy.reset_states()

    def load_weights(self, weights_path):
        if weights_path is None or not os.path.exists(weights_path):
            return
        self.model.load_weights(weights_path)

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.loss_func(labels, predictions) + self.model.losses
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.loss(loss)
        self.accuracy(labels, predictions)
        return {"loss": self.loss.result(), "accuracy": self.accuracy.result()}

    @tf.function
    def test_step(self, images, labels):
        predictions = self.model(images, training=False)
        loss = self.loss_func(labels, predictions)

        # import pdb
        # pdb.set_trace()
        self.loss(loss)
        self.accuracy(labels, predictions)
        return {"loss": self.loss.result(), "accuracy": self.accuracy.result()}

    @tf.function
    def predict(self, images):
        predictions = self.model(images)
        return predictions

    def train(self):
        """customized training function (alternative to model.fit())
        """
        self.model.compile(self.optimizer, self.loss_func, metrics=["accuracy"])
        self._reset_state_of_metrix_containers()

        train_steps_per_epoch = math.ceil(self.train_gen.samples / self.train_gen.batch_size)
        val_steps_per_epoch = math.ceil(self.val_gen.samples / self.val_gen.batch_size)
        self.callbacks.on_train_begin({
            "epochs": self.cfg["train"]["epochs"],
            "steps_per_epochs": train_steps_per_epoch
        })

        for epoch in range(self.start_epoch, self.cfg["train"]["epochs"]):
            self.callbacks.on_epoch_begin(epoch)
            self._reset_state_of_metrix_containers()
            for idx, (images, labels) in enumerate(self.train_gen):
                # if idx % 100 == 0:
                #     import pdb
                #     pdb.set_trace()
                if train_steps_per_epoch == idx:
                    break
                self.callbacks.on_train_batch_begin(idx)
                logs = self.train_step(images, labels)
                self.callbacks.on_train_batch_end(idx, logs)
            epoch_logs = copy.copy(logs)

            self.callbacks.on_test_begin()
            self._reset_state_of_metrix_containers()
            for idx, (images, labels) in enumerate(self.val_gen):
                if val_steps_per_epoch == idx:
                    break
                self.callbacks.on_test_batch_begin(idx)
                logs = self.test_step(images, labels)
                self.callbacks.on_test_batch_end(idx, logs)
            logs = tf_utils.to_numpy_or_python_type(logs)
            val_logs = {'val_' + name: val for name, val in logs.items()}
            self.callbacks.on_test_end(val_logs)

            epoch_logs.update(val_logs)
            self.callbacks.on_epoch_end(epoch, epoch_logs)
        self.callbacks.on_train_end(epoch_logs)

    # def test(self, test_gen, weight_path=None):
    #     if self.optimizer and self.loss_func:
    #         self.model.compile(self.optimizer, self.loss_func, metrics=["accuracy"])
    #     if weight_path:
    #         self.model.load_weights(weight_path)
    #     loss, acc = self.model.evaluate(test_gen)
    #     print("test data loss = {:.2f}, acc = {:.4f}".format(loss, acc))
