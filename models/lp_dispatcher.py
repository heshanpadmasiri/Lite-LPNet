import tensorflow as tf
from tensorflow.keras import datasets, layers, models,Input

from models.lp_2 import create_model as create_model_2
from models.lp_3 import create_model as create_model_3
from models.lp_4 import create_model as create_model_4

CONSTRUCTORS = {
    'v2' : create_model_2,
    'v3' : create_model_3,
    'v4' : create_model_4
}

def create_model(model_name):
    model_name = model_name.lower()
    return CONSTRUCTORS[model_name]()

def create_combined_model(model_name,input_shape, chars=7):
    """
    Use to create the combined model directly using the individual model.
    """
    image = Input(shape=input_shape,name='img')
    submodels = [create_model(model_name) for i in range(chars)]
    seperate_outputs = [model(image) for  model in submodels]
    output = tf.stack(seperate_outputs,axis=1)
    output = tf.math.argmax(output,axis=2)
    model = tf.keras.Model(inputs=image, outputs=output, name='combined_model')
    return model
