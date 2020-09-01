import argparse
import sys
from models.bbox_dispatcher import create_model
import bbox_to_tflite
import train_bbox

def stage_1_without_training(model_name):
    model = create_model(model_name,train_bbox.TARGET_SIZE + (3,))
    tf_lite_model = bbox_to_tflite.convert_model(model)
    bbox_to_tflite.save_model(tf_lite_model, model_name)

def stage_1_with_training(model_name, dataset_path):
    train_bbox.train(train_bbox.TARGET_SIZE,dataset_path,model_name)
    model = bbox_to_tflite.load_model(model_name)
    tf_lite_model = bbox_to_tflite.convert_model(model)
    bbox_to_tflite.save_model(tf_lite_model, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Directly create tflite model')
    parser.add_argument('stage', type=int, help='Stage 1 or Stage 2')
    parser.add_argument('model_name', type=str, help='Name of the model')
    parser.add_argument('-skip_train', type=bool, help='Skip the training step')
    parser.add_argument('-dataset_path', metavar='path', type=str, help="Path to dataset")
    args = parser.parse_args()
    if (not args.skip_train) and (not args.dataset_path):
        print("Can not train dataset not given")
        sys.exit(1)
    if args.stage == 1:
        if args.skip_train:
            stage_1_without_training(args.model_name)
        else:
            stage_1_with_training(args.model_name, args.dataset_path)