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

from ..utils import ConvBn2D

import logging

logger = logging.getLogger("FNet")
logger.setLevel(logging.INFO)


def init_pytorch(shape, dtype=tf.float32,):
    fan = np.prod(shape[:-1])
    bound = 1 / math.sqrt(fan)
    return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)


class ResBlk(tf.keras.Model):
    def __init__(self, c_out, pool, res=False, kernel_initializer=init_pytorch,
                 residual_dropout=0.0, spatial_dropout=0.0, use_depthwise_conv=False):
        super().__init__()
        bn = tf.keras.layers.BatchNormalization
        self.conv_bn = ConvBn2D(c_out, bn=bn, spatial_dropout=spatial_dropout,kernel_initializer=kernel_initializer)
        self.pool = pool
        self.res = res
        self.residual_dropout = residual_dropout
        if self.res:
            self.res1 = ConvBn2D(c_out, bn=bn, use_depthwise_conv=use_depthwise_conv)
            self.res2 = ConvBn2D(c_out, bn=bn, use_depthwise_conv=use_depthwise_conv)

    def call(self, inputs):
        h = self.pool(self.conv_bn(inputs))
        if self.res:
            p_1 = tf.random.uniform([] ,minval=0 ,maxval=1 ,dtype=tf.dtypes.float32,)
            # cond = tf.keras.backend.less(p_1 ,self.residual_dropout)
            # cond = tf.broadcast_to(cond ,[inputs.shape[0]])
            # h = tf.keras.backend.switch(cond ,h ,h + self.res2(self.res1(h)))

            if p_1 >= self.residual_dropout:
                h = h + self.res2(self.res1(h))
            else:
                pass
        return h


class FNet(tf.keras.Model):
    def __init__(self, start_kernels=64, weight=0.125, use_depthwise_conv=False,
                 enable_skip=False, enable_pool_before_skip=False,
                 residual_dropout=0.0, spatial_dropout=0.0):
        super().__init__()
        c = start_kernels
        pool = tf.keras.layers.MaxPooling2D()
        self.init_conv_bn = ConvBn2D(c, kernel_size=3)
        self.enable_skip = enable_skip
        self.enable_pool_before_skip = enable_pool_before_skip
        if enable_skip:
            self.skip = ConvBn2D(c * 2, kernel_size=1, strides=1,)
            self.avg_pool = tf.keras.layers.AveragePooling2D()
            self.max_pool = pool

        self.blk1 = ResBlk(c * 2, pool, res=True, use_depthwise_conv=use_depthwise_conv,
                           residual_dropout=residual_dropout, spatial_dropout=spatial_dropout, )

        self.blk2 = ResBlk(c * 4, pool, use_depthwise_conv=use_depthwise_conv,
                           residual_dropout=residual_dropout, spatial_dropout=spatial_dropout, )

        self.blk3 = ResBlk(c * 8, pool, res=True, use_depthwise_conv=use_depthwise_conv,
                           residual_dropout=residual_dropout, spatial_dropout=spatial_dropout, )

        self.pool = tf.keras.layers.GlobalMaxPool2D()
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.linear = tf.keras.layers.Dense(10, kernel_initializer=init_pytorch, use_bias=False)
        self.weight = weight
        self.concat = tf.keras.layers.Concatenate()

    def call(self, x, y):
        h = self.init_conv_bn(x)

        h = self.blk1(h)

        if self.enable_skip:
            if self.enable_pool_before_skip:
                k = self.pool(self.skip(self.avg_pool(h)))
            else:
                k = self.pool(self.skip(h))

        h = self.blk2(h)
        h = self.blk3(h)
        h = self.pool(h)
        if self.enable_skip:
            h = self.concat([h, k])
        h = self.linear(h) * self.weight
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h, labels=y)
        loss = tf.reduce_sum(ce)
        correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(h, axis=1), y), tf.float32))
        return loss, correct