import joblib
from pathlib import Path
import argparse
import os
import datetime
from dataset.bbox_dataset import get_datasets, get_kfold_dataset
from dataset.create_kfolds import create_kfolds
from models.bbox_dispatcher import create_model
from metrices.iou import IoU
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import utils

TRAIN = 0.7
VAL = 0.15
TEST = 0.15
BATCH_SIZE = 32
TARGET_SIZE = (480, 480)


def __get_paths__(model_name):
    log_dir = f"logs/simple_bbox/{model_name}/" + datetime.datetime.now(
    ).strftime("%y%m%d-%h%m%s")
    checkpoint_dir = f"checkpoints/{model_name}"
    return log_dir, checkpoint_dir


def __get_callbacks__(model_name):
    log_dir, checkpoint_dir = __get_paths__(model_name)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_iou',
        min_delta=0,
        patience=10,
        verbose=0,
        mode='max',
        baseline=None,
        restore_best_weights=True)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir, save_weights_only=True, monitor='val_loss')
    return [
        tensorboard_callback, early_stopping_callback,
        model_checkpoint_callback
    ]


def kfold_train(input_shape, dataset_path, model_name, folds, cache_dataset,
                starting_fold):
    fold_data = create_kfolds(Path(dataset_path), folds)
    for fold in range(starting_fold, folds):
        print(f'Training fold: {fold}')
        model = create_model(model_name, input_shape + (3, ))
        train_dataset, test_dataset = get_kfold_dataset(fold_data,
                                                        fold,
                                                        input_shape,
                                                        BATCH_SIZE,
                                                        cached=cache_dataset)
        callbacks = __get_callbacks__(model_name)
        model.fit(train_dataset,
                  validation_data=test_dataset,
                  verbose=2,
                  callbacks=[callbacks],
                  epochs=60)
        model.save(f'saved_models/simple_bbox/{model_name}_{fold}')
        val_eval = model.evaluate(x=test_dataset, return_dict=True)
        joblib.dump(val_eval,
                    f'saved_models/simple_bbox/{model_name}_eval.pkl')
        utils.zip_dir(f'fold_{fold}_saved_models.zip', 'saved_models')
        utils.zip_dir(f'fold_{fold}_log.zip', 'logs')


def train(input_shape, dataset_path, model_name, restore=False):
    dataset_size = len(os.listdir(dataset_path))
    train_size = int(TRAIN * dataset_size)
    test_size = int(TEST * dataset_size)
    val_size = int(VAL * dataset_size)
    train_dataset, val_dataset, test_dataset = get_datasets(
        os.path.join(dataset_path, '*'),
        train_size,
        val_size,
        test_size,
        target_size=input_shape,
        batch_size=BATCH_SIZE)

    model = create_model(model_name, input_shape + (3, ))
    if restore:
        _, checkpoint_path = __get_paths__(model_name)
        model.load_weights(checkpoint_path)
        print(f"model weights from {checkpoint_path} restored")
    callbacks = __get_callbacks__(model_name)
    model.fit(train_dataset,
              validation_data=test_dataset,
              verbose=2,
              callbacks=[callbacks],
              epochs=60)
    model.save(f'saved_models/simple_bbox/{model_name}')
    val_eval = model.evaluate(x=val_dataset, return_dict=True)
    joblib.dump(val_eval, f'saved_models/simple_bbox/{model_name}_eval.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train bbox model")
    parser.add_argument('dataset_path',
                        metavar='path',
                        type=str,
                        help='Path to ccpd dataset')
    parser.add_argument('model_name', type=str, help='name for the model')
    parser.add_argument('-input_shape', type=int, help='input image size')
    parser.add_argument('-restore_weights',
                        type=bool,
                        help='restore presaved checkpoints')
    parser.add_argument('-mixed_precision',
                        type=bool,
                        help='use mixed precision training')
    parser.add_argument(
        '-folds',
        type=int,
        help=
        'Number of folds for kfold validation. Leave empty to ignore kfolds')
    parser.add_argument(
        '-starting_fold',
        type=int,
        help=
        'Set the starting fold. Default to 0. Must define folds to use this')
    parser.add_argument('-cache_dataset',
                        type=bool,
                        help='Cache dataset to disk. Default false')
    args = parser.parse_args()
    if args.mixed_precision:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
    if not args.input_shape:
        input_shape = TARGET_SIZE
    else:
        input_shape = (args.input_shape, ) * 2
    model_name = args.model_name
    if not args.folds:
        if args.restore_weights:
            train(input_shape, args.dataset_path, model_name, True)
        else:
            train(input_shape, args.dataset_path, model_name)
    else:
        cache = True if args.cache_dataset else False
        starting_fold = args.starting_fold if args.starting_fold else 0
        assert 0 <= starting_fold < args.folds
        kfold_train(input_shape, args.dataset_path, model_name, args.folds,
                    cache, starting_fold)
