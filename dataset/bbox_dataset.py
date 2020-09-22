import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import datasets
from dataset.create_kfolds import create_kfolds

ORIGINAL_SIZE = (1160, 720)


@tf.function
def extract_bounding_box(file_name, target_size):
    x_factor = target_size[1] / ORIGINAL_SIZE[1]
    y_factor = target_size[0] / ORIGINAL_SIZE[0]

    data = tf.strings.split(file_name, '-')
    bounding_box_data = tf.gather_nd(data, [2])
    pairs = tf.strings.split(bounding_box_data, '_')
    p_1 = tf.gather_nd(pairs, [0])
    p_2 = tf.gather_nd(pairs, [1])

    tmp = tf.strings.split(p_1, '&')
    x_1 = tf.strings.to_number(tf.gather_nd(tmp, [0])) * x_factor
    y_1 = tf.strings.to_number(tf.gather_nd(tmp, [1])) * y_factor

    tmp = tf.strings.split(p_2, '&')
    x_2 = tf.strings.to_number(tf.gather_nd(tmp, [0])) * x_factor
    y_2 = tf.strings.to_number(tf.gather_nd(tmp, [1])) * y_factor

    x = (x_1 + x_2) / 2
    y = (y_1 + y_2) / 2
    w = tf.abs((x_2 - x_1))
    h = tf.abs((y_1 - y_2))

    x /= target_size[0]
    w /= target_size[0]

    y /= target_size[1]
    h /= target_size[1]

    return tf.stack([x, y, w, h])


@tf.function
def decode_img(img, target_size):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, target_size)


def getPathProcessor(target_size):
    @tf.function
    def process_path(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        file_name = tf.strings.split(parts[-1], '.')[0]
        length = tf.strings.length(file_name)
        file_name = tf.strings.substr(file_name, 0, length - 4)
        bounding_box = extract_bounding_box(file_name, target_size)
        img = tf.io.read_file(file_path)
        img = decode_img(img, target_size)
        return img, bounding_box

    return process_path


def get_datasets(data_path,
                 train_size,
                 val_size,
                 test_size,
                 target_size=(480, 480),
                 batch_size=32):
    file_name_ds = tf.data.Dataset.list_files(data_path).prefetch(
        tf.data.experimental.AUTOTUNE)
    process_path = getPathProcessor(target_size)
    bbox_ds = file_name_ds.map(
        process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = bbox_ds.take(train_size).batch(batch_size)
    test_dataset = bbox_ds.skip(train_size).batch(batch_size)
    val_dataset = bbox_ds.skip(val_size).batch(batch_size)
    test_dataset = bbox_ds.take(test_size).batch(batch_size)
    return train_dataset, val_dataset, test_dataset


def get_kfold_dataset(fold_data,
                      fold,
                      target_size=(480, 480),
                      batch_size=32,
                      cached=False):
    train_data = fold_data[fold]['train']
    test_data = fold_data[fold]['test']
    train_file_name_ds = tf.data.Dataset.from_tensor_slices(
        list(map(lambda file_path: str(file_path), train_data)))
    test_file_name_ds = tf.data.Dataset.from_tensor_slices(
        list(map(lambda file_path: str(file_path), test_data)))
    process_path = getPathProcessor(target_size)
    train_dataset = train_file_name_ds.map(
        process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_file_name_ds.map(
        process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if cached:
        train_cache = f'./cache/train_{fold}'
        test_cache = f'./cache/test_{fold}'
        train_dataset = train_dataset.cache(train_cache)
        test_dataset = test_dataset.cache(test_cache)
    train_dataset = train_dataset.batch(batch_size).prefetch(
        tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(
        tf.data.experimental.AUTOTUNE)
    return train_dataset, test_dataset
