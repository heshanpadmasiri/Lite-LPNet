import argparse
import sys
from models.bbox_dispatcher import create_model as bbox_create_model
from models.lp_dispatcher import create_model as lp_create_model
from models.lp_dispatcher import create_combined_model
import model_utils
import train_bbox
import train_lp

def stage_1_without_training(model_name):
    model = bbox_create_model(model_name,train_bbox.TARGET_SIZE + (3,))
    tf_lite_model = model_utils.convert_model(model)
    model_utils.save_model(tf_lite_model, f's1_{model_name}')

def stage_1_with_training(model_name, dataset_path):
    train_bbox.train(train_bbox.TARGET_SIZE,dataset_path,model_name)
    model = model_utils.load_model_bbox(model_name)
    tf_lite_model = model_utils.convert_model(model)
    model_utils.save_model(tf_lite_model, f's1_{model_name}')


def stage_2_without_training_combined(model_name):
    model = create_combined_model(model_name, train_lp.TARGET_SIZE + (3,))
    tf_lite_model = model_utils.convert_model(model)
    model_utils.save_model(tf_lite_model, f's2_{model_name}')

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

    if args.stage == 2:
        if args.skip_train:
            stage_2_without_training_combined(args.model_name)