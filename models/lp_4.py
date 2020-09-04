import tensorflow as tf
from tensorflow.keras import layers, models, Input
from models.nn_blocks import rpnet_block_3_seq

def create_model():
    model = models.Sequential()
    for i in range(4):
        model.add(rpnet_block_3_seq())
        model.add(layers.BatchNormalization(axis=-1,
                                  epsilon=1e-3,
                                  momentum=0.999))
    model.add(layers.Flatten())
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dense(35, activation='softmax'))
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model