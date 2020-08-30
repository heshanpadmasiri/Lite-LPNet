import tensorflow as tf
from tensorflow.keras import layers, models, Input
from models.nn_blocks import rpnet_block_2

def create_model():
    model = models.Sequential()
    filters = [(48,64), (64,128), (128,160)]
    for i in range(2):
        model.add(rpnet_block_2(filters[i]))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(35, activation='softmax'))
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model