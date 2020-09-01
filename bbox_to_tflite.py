import tensorflow as tf
import argparse


def load_model(model_name):
    model = tf.keras.models.load_model(f'saved_models/simple_bbox/{model_name}')
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
    model = load_model(args.model_name)
    tflite_model = convert_model(model)
    save_model(tflite_model, args.model_name) 
