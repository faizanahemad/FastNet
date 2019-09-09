from tensorflow.keras.layers import BatchNormalization, Layer,InputSpec
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.initializers import Initializer
from tensorflow.python.keras.backend import _regular_normalize_batch_in_training
import tensorflow.keras.backend as K
import tensorflow
import tensorflow as tf
import math
import gc
import numpy as np

def init_pytorch(shape, dtype=tf.float32, partition_info=None):
    fan = np.prod(shape[:-1])
    bound = 1 / math.sqrt(fan)
    return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)