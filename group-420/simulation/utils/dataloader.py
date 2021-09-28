from pathlib import Path
import tensorflow as tf


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)

    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.per_image_standardization(img)

    return img


def process_path(file_path):

    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)

    return img, tf.strings.to_number(tf.strings.split(tf.strings.split(file_path, '_')[-1], '.')[0]) / 1000.0
