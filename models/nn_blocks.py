import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input

def mobilenet_block_2(inputs):
    x = layers.Conv2D(3,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None)(inputs)
    x = layers.BatchNormalization(axis=-1,
                                  epsilon=1e-3,
                                  momentum=0.999)(inputs)
    x = layers.ReLU(6.)(x)    

    x = layers.DepthwiseConv2D(kernel_size=3,
                               strides=2,
                               activation=None,
                               use_bias=False,
                               padding='valid')(x)
    x = layers.BatchNormalization(axis=-1,
                                  epsilon=1e-3,
                                  momentum=0.999)(x)
    x = layers.ReLU(6.)(x)
    x = layers.Conv2D(3,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None)(x)
    x = layers.BatchNormalization(axis=-1,
                                  epsilon=1e-3,
                                  momentum=0.999)(x)
    return x
    
def mobilenet_block_1(inputs):
    x = layers.Conv2D(3,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None)(inputs)
    x = layers.BatchNormalization(axis=-1,
                                  epsilon=1e-3,
                                  momentum=0.999)(x)
    x = layers.ReLU(6.)(x)    

    x = layers.DepthwiseConv2D(kernel_size=3,
                               strides=1,
                               activation=None,
                               use_bias=False,
                               padding='same')(x)
    x = layers.BatchNormalization(axis=-1,
                                  epsilon=1e-3,
                                  momentum=0.999)(x)
    x = layers.ReLU(6.)(x)
    x = layers.Conv2D(3,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None)(x)
    x = layers.BatchNormalization(axis=-1,
                                  epsilon=1e-3,
                                  momentum=0.999)(x)
    return layers.Add()([inputs, x])
