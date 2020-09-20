import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input
from models.nn_blocks import rpnet_block_1, rpnet_block_3
from metrices.iou import IoUMetric


def __get_model__(feature_extractor, input_shape, extractor_layer=None):
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(4)
    model = tf.keras.Sequential(
        [feature_extractor, global_average_layer, prediction_layer])

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=[IoUMetric(input_shape)])
    return model


def __get_backborn__(input_size):
    image = Input(shape=input_size, name='img')
    x = rpnet_block_3(image)
    x = rpnet_block_3(x)
    x = rpnet_block_3(x)
    x = rpnet_block_1(x)
    x = rpnet_block_1(x)
    x = rpnet_block_1(x)
    backborn = tf.keras.Model(inputs=image, outputs=x)
    return backborn


def create_model(input_shape):
    backborn = __get_backborn__(input_shape)
    model = __get_model__(backborn, input_shape)
    return model
