import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
import os

import logging
logger = logging.getLogger("Image Augmentations")
logger.setLevel(logging.INFO)


def replace_slice(input_: tf.Tensor, replacement, begin) -> tf.Tensor:
    inp_shape = tf.shape(input_)
    size = tf.shape(replacement)
    padding = tf.stack([begin, inp_shape - (begin + size)], axis=1)
    replacement_pad = tf.pad(replacement, padding)
    mask = tf.pad(tf.ones_like(replacement, dtype=tf.bool), padding)
    return tf.where(mask, replacement_pad, input_)



def get_tf_batch_cutout_eraser(minimum,maximum, area: int = 81, c: int = 3, min_aspect_ratio=0.5, max_aspect_ratio=1/0.5):
    sqrt_area = np.sqrt(area)
    def tf_cutout(x: tf.Tensor) -> tf.Tensor:
        """
        Cutout data augmentation. Randomly cuts a h by w whole in the image, and fill the whole with zeros.
        :param x: Input image.
        :param h: Height of the hole.
        :param w: Width of the hole
        :param c: Number of color channels in the image. Default: 3 (RGB).
        :return: Transformed image.
        """
        dtype = x.dtype
        minval = tf.cast(minimum, dtype=dtype)
        maxval = tf.cast(maximum, dtype=dtype)

        aspect_ratio = tf.random.uniform([],min_aspect_ratio,max_aspect_ratio)
        h = int(area/aspect_ratio)
        w = int(area*aspect_ratio)

        shape = tf.shape(x)
        x0 = tf.random.uniform([], 0, shape[1] + 1 - h, dtype=dtype)
        y0 = tf.random.uniform([], 0, shape[2] + 1 - w, dtype=dtype)

        slic = tf.random.uniform([shape[0], h, w, c], minval=minval, maxval=maxval, dtype=dtype)
        x = replace_slice(x, slic, [0, x0, y0, 0])
        return x
    return tf_cutout

def get_hue_aug(max_delta):
    return lambda x: tf.image.random_hue(x,max_delta=max_delta)

def get_brightness_aug(max_delta):
    return lambda x: tf.image.random_brightness(x,max_delta=max_delta)

def get_contrast_aug(max_delta):
    return lambda x: tf.image.random_contrast(x,max_delta=max_delta)

def get_saturation_aug(lower,upper):
    return lambda x: tf.image.random_saturation(x,lower=lower,upper=upper)

def get_random_pad_crop(pad_height,pad_width, cropped_height,cropped_width):
    def transformer(x):
        shape = tf.shape(x)
        x = tf.image.resize_with_crop_or_pad(x, pad_height, pad_width)
        tf.image.random_crop(x, [shape[0], cropped_height, cropped_width, shape[3]])
    return transformer

def x_y_wrapper(fn, x: tf.Tensor, y: tf.Tensor):
    return fn(x),y

