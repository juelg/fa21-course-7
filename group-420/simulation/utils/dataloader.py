import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

FLIP_PERCENTAGE = 0.5
BLUR_PERCENTAGE = 0.1


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)

    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.per_image_standardization(img)

    return img


def apply_flip(img, steering_angle):
    def flip_true_fn(): return tf.image.flip_left_right(img), -steering_angle

    def flip_false_fn(): return img, steering_angle

    val = tf.random.uniform((), dtype=tf.dtypes.float32)
    return tf.cond(val < FLIP_PERCENTAGE, true_fn=flip_true_fn, false_fn=flip_false_fn)


def apply_gaus_blur(img):
    blur_size = np.random.randint(2, 4)
    blur_sigma = np.random.rand() * 0.5
    val = tf.random.uniform((), dtype=tf.dtypes.float32)

    def gaus_blur_true_fn(): return tfa.image.gaussian_filter2d(img, blur_size, blur_sigma, "REFLECT", 0)

    def gaus_blur_false_fn(): return img

    return tf.cond(val < BLUR_PERCENTAGE, true_fn=gaus_blur_true_fn, false_fn=gaus_blur_false_fn)


def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)

    steering_angle = tf.strings.to_number(tf.strings.split(tf.strings.split(file_path, '_')[-1], '.')[0]) / 1000.0
    speed = tf.strings.to_number(tf.strings.split(file_path, '_')[-2]) / 1000.0

    img, steering_angle = apply_flip(img, steering_angle)

    img = apply_gaus_blur(img)

    return img, steering_angle, speed

