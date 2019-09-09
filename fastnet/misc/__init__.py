from tensorflow.keras.backend import set_session
from tensorflow.keras.backend import clear_session
from tensorflow.keras.backend import get_session
import tensorflow
import tensorflow as tf
import gc

# Reset Keras Session
def reset_keras(config_builder=None):
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    _ = gc.collect() # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    # config = tensorflow.ConfigProto(log_device_placement=True, allow_soft_placement=True,)
    config = tf.compat.v1.ConfigProto() if config_builder is None else config_builder()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.Session(config=config))