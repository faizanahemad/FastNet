import numpy as np
import tensorflow as tf
import os

from tensorflow.keras.layers import BatchNormalization, Layer,InputSpec
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.initializers import Initializer
from tensorflow.python.keras.backend import _regular_normalize_batch_in_training
import tensorflow.keras.backend as K
import tensorflow
import tensorflow as tf
import math

import gc

import logging

logger = logging.getLogger("Conv Utils")
logger.setLevel(logging.INFO)


class ConvBn2D(tf.keras.Model):
    def __init__(self, c_out, kernel_size=3,
                 bn=tf.keras.layers.BatchNormalization, epsilon=1e-5,
                 strides=1, activation=tf.nn.relu, spatial_dropout=0.0,
                 use_depthwise_conv=False,kernel_initializer='glorot_uniform'):
        super().__init__()
        if use_depthwise_conv:
            tf.keras.layers.SeparableConv2D(filters=c_out, depth_multiplier=2, kernel_size=kernel_size,
                                            strides=strides, padding="SAME", kernel_initializer=kernel_initializer,
                                            use_bias=False)
        else:
            self.conv = tf.keras.layers.Conv2D(filters=c_out, kernel_size=kernel_size,
                                               strides=strides, padding="SAME", kernel_initializer=kernel_initializer,
                                               use_bias=False)
        self.activation = activation
        self.spatial_dropout = spatial_dropout
        self.sd = tf.keras.layers.SpatialDropout2D(self.spatial_dropout)

        assert bn is not None
        self.bn = bn(momentum=0.9, epsilon=epsilon)

    def call(self, inputs):
        res = self.bn(self.conv(inputs))

        if self.spatial_dropout > 0:
            res = self.sd(res)

        if self.activation is not None:
            res = self.activation(res)
        return res