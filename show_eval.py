from pathlib import Path
import argparse
import joblib

saved_model_path = Path('./saved_models')
bbox_path = saved_model_path/'simple_bbox'
lp_path = saved_model_path/'lp_seperate'

def __get_pickle_files__(base_path:Path):
    return [each for each in base_path.iterdir() if each.suffix == '.pkl']

def get_eval_files(base_path:Path):
    pickle_files = __get_pickle_files__(base_path)
    eval_files = [each for each in pickle_files if 'iou' in each.name]
    return eval_files

def print_eval_results(eval_path:Path):
    name = eval_path.name.strip('.pkl').split('_')[0]
    data = joblib.load(eval_path)
    print(f'{name} : {data[1]}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train bbox model")
    parser.add_argument('stage', type=int, help='stage to show eval results')
    args = parser.parse_args()
    stage = args.stage
    if stage == 1:
        eval_results = get_eval_files(bbox_path)
        for each in eval_results:
            print_eval_results(each)