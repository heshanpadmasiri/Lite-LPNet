import argparse
import joblib
import os
from pathlib import Path
import itertools


def __get_files__(path: Path):
    files = sorted(list(path.glob('*.jpg')))
    return files


def create_kfolds(dataset_path: Path, folds: int):
    files = __get_files__(dataset_path)
    number_files = len(files)
    fold_size = len(files) // folds
    print(f'dataset directory:{dataset_path}')
    print(f'number of files: {number_files} fold size : {fold_size}')
    fold_files = [
        files[i * fold_size:(i + 1) * fold_size] for i in range(folds)
    ]
    data = []
    for i in range(folds):
        train_files = [fold_files[j] for j in range(folds) if j != i]
        train_files = list(itertools.chain(*train_files))
        test_files = fold_files[i]
        data.append({'train': train_files, 'test': test_files})
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create kfold datasets")
    parser.add_argument('dataset_path',
                        metavar='path',
                        type=str,
                        help='Path to ccpd dataset')
    parser.add_argument('folds', type=int, help='Number of folds')
    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise ValueError("Invalid dataset path")
    create_kfolds(dataset_path, args.folds)
