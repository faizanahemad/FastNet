import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
import os

import logging
logger = logging.getLogger("TFRecord Helpers")
logger.setLevel(logging.INFO)

def byte_to_tf_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_to_tf_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def int64_to_tf_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def build_serializer(feature_names,dtypes_converter):
    def serialize_example(*args):
        feature = {feature:dtypes_converter[feature](args[i]) for i,feature in enumerate(feature_names)}
        proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return proto.SerializeToString()
    return serialize_example

def build_tf_serializer(feature_names,dtypes_converter):
    serializer = build_serializer(feature_names,dtypes_converter)
    def tf_serialize_example(*args):
        tf_string = tf.py_function(serializer,args,tf.string)
        return tf.reshape(tf_string, ())
    return tf_serialize_example

def store_numpy_arrays_as_tfrecord(numpy_arrays,filename,feature_names,dtypes_converter, compression_type=None):
    assert type(numpy_arrays) == tuple or type(numpy_arrays) == list
    ds = tf.data.Dataset.from_tensor_slices(numpy_arrays)
    store_dataset_as_tfrecord(ds,filename,feature_names,dtypes_converter, compression_type=compression_type)

def store_dataset_as_tfrecord(dataset,filename,feature_names,dtypes_converter, compression_type=None):
    dataset = dataset.map(build_tf_serializer(feature_names,dtypes_converter))
    writer = tf.data.experimental.TFRecordWriter(filename, compression_type=compression_type)
    logger.debug("Storing to: %s",filename)
    writer.write(dataset)


def read_tfrecord_as_dataset(filename, feature_description, batch_size,
                        shuffle=True, shuffle_buffer_size=10000, compression_type=None):
    ds = tf.data.TFRecordDataset(filename,compression_type=compression_type)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    parser = lambda x: tf.io.parse_example(x, feature_description)
    ds = ds.map(parser)
    return ds



def get_cifar10(dir, batch_size, shuffle=True, shuffle_buffer_size=10000, compression_type=None):
    # check if cifar10 tfrecord exists locally, just check file exists
    # if it does then get it from tfrecord
    # else download and store the tfrecord and then get it from tfreccord
    if not os.path.exists(dir):
        os.makedirs(dir)
    train_loc = os.path.join(dir,"cifar10.train.tfrecords")
    test_loc = os.path.join(dir, "cifar10.test.tfrecords")
    cifar10_exists = os.path.exists(train_loc) and os.path.exists(test_loc)

    if not cifar10_exists:


        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        feature_names = ["image", "label"]
        dtypes_converter = {"image":lambda img:byte_to_tf_feature(bytes(img)),"label":lambda label:int64_to_tf_feature(label)}


        store_numpy_arrays_as_tfrecord((x_train, y_train), train_loc, feature_names=feature_names,
                                  dtypes_converter=dtypes_converter, compression_type=compression_type)
        store_numpy_arrays_as_tfrecord((x_test, y_test), test_loc, feature_names=feature_names,
                                  dtypes_converter=dtypes_converter, compression_type=compression_type)

    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64)
    }
    train = read_tfrecord_as_dataset(train_loc,feature_description, batch_size, shuffle, shuffle_buffer_size, compression_type)
    test = read_tfrecord_as_dataset(test_loc,feature_description, batch_size, shuffle, shuffle_buffer_size, compression_type)

    def parser(x):
        labels = tf.map_fn(lambda y: tf.cast(y, tf.int64), x['label'])
        imgs = tf.map_fn(lambda y:tf.cast(tf.io.decode_raw(y, tf.uint8),tf.float32), x['image'],dtype=tf.float32)
        return imgs,labels

    train = train.map(parser).prefetch(tf.data.experimental.AUTOTUNE)
    test = test.map(parser).prefetch(tf.data.experimental.AUTOTUNE)
    return train,test




def get_cifar100():
    pass

def get_imagenet():
    pass

def get_mnist():
    pass

print(tf.__version__)
get_cifar10(os.path.join(os.getcwd(),"cifar10/"),2)

# get_cifar10(os.getcwd(),2)


