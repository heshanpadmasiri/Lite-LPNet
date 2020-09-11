import argparse
import model_utils
import tensorflow as tf
from metrices.iou import IoU
from dataset.bbox_dataset import get_datasets as get_bbox_dataset
import os
import joblib

TRAIN = 0.7
VAL = 0.15
TEST = 0.15
BATCH_SIZE = 32

def __load_model__(stage, model_name):
    if stage == 1:
        model = model_utils.load_model_bbox(model_name)
    else:
        raise NotImplementedError("Model loading for lp models not implemented")
    return model

def __get_dataset__(stage, dataset_path, input_shape):
    dataset_size = len(os.listdir(dataset_path))
    train_size = int(TRAIN * dataset_size)
    test_size = int(TEST * dataset_size)
    val_size = int(VAL * dataset_size)
    if stage == 1:
        train_dataset, val_dataset, test_dataset = get_bbox_dataset(os.path.join(dataset_path,'*'), train_size, val_size, test_size, target_size=input_shape, batch_size=BATCH_SIZE)
    else:
        raise NotImplementedError("Datasets for stage 2 are not implemented")
    return val_dataset

def __evaluate_stage_1__(model:tf.keras.Model,input_shape:tuple,dataset):
    eval = model.evaluate(x=dataset, return_dict=True)
    correct, ap, ious = IoU(dataset, model, input_shape)
    eval['IoU'] = {
        'correct': correct,
        'AP': ap,
        'IoUs': ious
    }
    print(f'AP : {ap}')
    return eval

def evaluate(model:tf.keras.Model, stage:int, dataset_path:str):
    input_shape = (model.input.shape[1], model.input.shape[2])
    dataset = __get_dataset__(stage, dataset_path, input_shape)
    if stage == 1:
        eval = __evaluate_stage_1__(model, input_shape, dataset)
    else:
        raise NotImplementedError("Stage 2 evaluation not implemented")
    return eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument('stage', type=int, help='Stage 1 or Stage 2')
    parser.add_argument('model_name', type=str, help='Name of the model')
    parser.add_argument('dataset_path', metavar='path', type=str, help="Path to dataset")
    parser.add_argument('-dump', type=bool, help='save eval to disk')
    args = parser.parse_args()
    stage = args.stage
    model_name = args.model_name
    dataset_path = args.dataset_path
    model = __load_model__(stage, model_name)
    eval = evaluate(model,stage,dataset_path)
    if(args.dump):
        joblib.dump(eval,f'eval_{stage}_{model_name}.pkl')