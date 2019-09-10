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


def get_cutout_eraser(minimum, maximum, area: int = 81, c: int = 3, min_aspect_ratio=0.5, max_aspect_ratio=1 / 0.5):
    sqrt_area = np.sqrt(area)

    def get_h_w(aspect_ratio):
        h = sqrt_area / aspect_ratio
        w = tf.math.round(area / h)
        h = tf.math.round(h)
        h = tf.cast(h, tf.int32)
        w = tf.cast(w, tf.int32)
        return h, w

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

        aspect_ratio = tf.random.uniform([], min_aspect_ratio, max_aspect_ratio)
        h, w = get_h_w(aspect_ratio)

        shape = tf.shape(x)
        x0 = tf.random.uniform([], 0, shape[1] + 1 - h, dtype=tf.int32)
        y0 = tf.random.uniform([], 0, shape[2] + 1 - w, dtype=tf.int32)

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


def get_hflip_aug():
    return lambda x: tf.image.random_flip_left_right(x)


def get_random_pad_crop(pad_height,pad_width, cropped_height,cropped_width):
    def transformer(x):
        shape = tf.shape(x)
        paddings = [[0,0],[pad_height,pad_height],[pad_width,pad_width],[0,0]]
        x = tf.pad(x,paddings,mode='REFLECT',)
        return tf.image.random_crop(x, [shape[0], cropped_height, cropped_width, shape[3]])
    return transformer


def get_first_argument_transformer(fn):
    def transformer_wrapper(x: tf.Tensor, *args):
        return tuple([fn(x)]+list(args))
    return transformer_wrapper


def combine_transformers(*transformers):
    def wrapper(*args):
        for t in transformers:
            args = t(*args)
        return tuple(args)
    return wrapper

