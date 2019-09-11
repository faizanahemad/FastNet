import tensorflow as tf


class BatchLRScheduler(tf.keras.callbacks.Callback):

    def __init__(self, lr_schedule, momentum_schedule, batch_size, total_samples, verbose=0):
        self.lr_schedule = lr_schedule
        self.momentum_schedule = momentum_schedule
        self.verbose = verbose
        self.batch_size = batch_size
        self.total_samples = total_samples
        self.batches_per_epoch = total_samples//batch_size + 1
        self.history = {}
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def plot(self):
        pass

