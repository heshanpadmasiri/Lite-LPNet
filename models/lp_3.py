import tensorflow as tf
from tensorflow.keras import layers, models, Input
from models.nn_blocks import rpnet_block_1_seq

def create_model():
    model = models.Sequential()
    for i in range(4):
        model.add(rpnet_block_1_seq())
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(35, activation='softmax'))
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model