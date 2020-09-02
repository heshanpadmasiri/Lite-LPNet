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

def rpnet_block_1(inputs,k=5):
    X = layers.Conv2D(64, (1, k), padding='same')(inputs)
    X = layers.Conv2D(64, (k, 1), padding='same', strides=2)(X)
    X = layers.MaxPool2D((2,2), strides=2)(X)
    return X

def rpnet_block_1_seq(k=5):
    block = models.Sequential()
    block.add(layers.Conv2D(64, (1,k), padding='same',activation='relu'))
    block.add(layers.Conv2D(64, (k,1), padding='same', strides=2, activation='relu'))
    block.add(layers.MaxPool2D((2,2), strides=2))
    return block
    
def rpnet_block_2(filters=[64,64]):
    block = models.Sequential()
    block.add(layers.Conv2D(filters[0], (5,5), strides=2,padding='same'))
    block.add(layers.MaxPool2D((2,2), strides=2))
    block.add(layers.Conv2D(filters[1], (5,5), strides=2, padding='same'))
    block.add(layers.MaxPool2D((2,2), strides=2))
    return block

def rpnet_block_3(inputs,k=5):
    X = layers.Conv2D(64, (1, k), padding='same')(inputs)
    X = layers.Conv2D(64, (k, 1), padding='same', strides=2)(X)
    return X