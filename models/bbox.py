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

def __get_backborn__(input_size,blocks=4):
    image = Input(shape=input_size,name='img')
    X = image
    for _ in range(blocks):
        X = layers.Conv2D(64,(1,5),padding='same')(X)
        X = layers.Conv2D(64,(5,1),strides=2, padding='same')(X)
        X = layers.MaxPool2D((2,2),strides=2)(X)
     
    backborn = tf.keras.Model(inputs=image, outputs=X)
    return backborn

def create_model(input_shape):
    backborn = __get_backborn__(input_shape)
    model = __get_model__(backborn)
    return model
