import os
import tensorflow as tf 
from tensorflow.keras import datasets


ORIGINAL_SIZE = (1160,720)

# dataset parameters
ORIGINAL_SIZE = (1160,720)


@tf.function
def extract_box_coord(file_name):
    x_factor = 1 / ORIGINAL_SIZE[1]
    y_factor = 1 / ORIGINAL_SIZE[0]

    data = tf.strings.split(file_name,'-')
    bounding_box_data = tf.gather_nd(data,[2])
    pairs = tf.strings.split(bounding_box_data,'_')
    p_1 = tf.gather_nd(pairs, [0])
    p_2 = tf.gather_nd(pairs, [1])

    tmp = tf.strings.split(p_1,'&')
    x_1 = tf.strings.to_number(tf.gather_nd(tmp, [0])) 
    y_1 = tf.strings.to_number(tf.gather_nd(tmp, [1])) 

    tmp = tf.strings.split(p_2,'&')
    x_2 = tf.strings.to_number(tf.gather_nd(tmp, [0])) 
    y_2 = tf.strings.to_number(tf.gather_nd(tmp, [1])) 

    x_low = tf.math.minimum(x_1,x_2)    
    y_high = tf.math.minimum(y_1,y_2)
    
    height = tf.math.abs(y_1 - y_2)
    width = tf.math.abs(x_1 - x_2)
    
    box = tf.stack([y_high,x_low,height,width])
    box = tf.cast(box, tf.int32)
    return box

@tf.function
def extract_license_plate(file_name):
    data = tf.strings.split(file_name,'-')
    license_plate_data = tf.gather_nd(data,[4])
    return tf.strings.to_number(tf.strings.split(license_plate_data,'_'))

@tf.function
def decode_img(img,target_size,box_coord):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    cropped_image = tf.image.crop_to_bounding_box(img, box_coord[0], box_coord[1], box_coord[2], box_coord[3])
    cropped_image = tf.image.resize(cropped_image,target_size)
    return cropped_image

def getPathProcessor(target_size,idx):
    @tf.function
    def process_path(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        file_name = tf.strings.split(parts[-1],'.')[0]
        length = tf.strings.length(file_name)
        file_name = tf.strings.substr(file_name,0,length-4)
        bbox_coord = extract_box_coord(file_name)
        license_plate = extract_license_plate(file_name)[idx]
        img = tf.io.read_file(file_path)
        img = decode_img(img,target_size,bbox_coord)
        return img, license_plate
    return process_path

def get_datasets(data_path,train_size, val_size, test_size, idx=0,target_size=(480,480), batch_size=32):
    file_name_ds = tf.data.Dataset.list_files(data_path).prefetch(tf.data.experimental.AUTOTUNE)
    process_path = getPathProcessor(target_size,idx)
    bbox_ds = file_name_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = bbox_ds.take(train_size).batch(batch_size,drop_remainder=True)
    test_dataset = bbox_ds.skip(train_size).batch(batch_size,drop_remainder=True)
    val_dataset = bbox_ds.skip(val_size).batch(batch_size,drop_remainder=True)
    test_dataset = bbox_ds.take(test_size).batch(batch_size,drop_remainder=True)
    return train_dataset, val_dataset, test_dataset