import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input

def __get_model__(feature_extractor,extractor_layer=None):
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(4)
    model = tf.keras.Sequential([
      feature_extractor,
      global_average_layer,
      prediction_layer
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def block_2(inputs):
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
    
def block_1(inputs):
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

def __get_backborn__(input_size):
    image = Input(shape=input_size,name='img')
    x = block_1(image)
    x = block_2(x)
    x = block_1(x)
    x = block_2(x)
    x = block_1(x)
    x = block_1(x)
    x = block_2(x)
    x = block_1(x)
    x = block_1(x)
    x = block_2(x)
    backborn = tf.keras.Model(inputs=image, outputs=x)
    return backborn

def create_model(input_shape):
    backborn = __get_backborn__(input_shape)
    model = __get_model__(backborn)
    return model
