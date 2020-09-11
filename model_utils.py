import tensorflow as tf
import argparse


def load_model_bbox(model_name):
    model = tf.keras.models.load_model(f'saved_models/simple_bbox/{model_name}')
    return model

def __load_model_lp_char__(model_name, idx):
    model = tf.keras.models.load_model(f'saved_models/lp_seperate/{model_name}/{idx}')
    model._name = f'char_{idx}'
    model.trainable = False
    return model

def __create_combined_model_lp__(seperate_models,input_shape):
    image = tf.keras.Input(shape=input_shape,name='img')
    seperate_outputs = [model(image) for  model in seperate_models]
    output = tf.stack(seperate_outputs,axis=1)
    output = tf.math.argmax(output,axis=2)
    model = tf.keras.Model(inputs=image, outputs=output, name='combined_model')
    return model

def load_model_lp(model_name):
    sub_models = [__load_model_lp_char__(model_name, idx) for idx in range(7)]
    input_shape = (sub_models[0].input.shape[1], sub_models[0].input.shape[2], sub_models[0].input.shape[3])
    model = __create_combined_model_lp__(sub_models, input_shape)
    return model

def convert_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    return tflite_model

def save_model(model, model_name):
    with tf.io.gfile.GFile(f'{model_name}.tflite','wb') as f:
        f.write(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert tf model to tflite')
    parser.add_argument('model_name', type=str, help='name of the model to convert')

    args = parser.parse_args()
    model = load_model_bbox(args.model_name)
    tflite_model = convert_model(model)
    save_model(tflite_model, args.model_name) 
