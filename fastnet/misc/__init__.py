import tensorflow
import tensorflow as tf
import gc

def msg(*args):
    assert len(args)>0
    message = ""
    for arg in args:
        arg = str(arg)
        message =message + " " + arg
    return message
