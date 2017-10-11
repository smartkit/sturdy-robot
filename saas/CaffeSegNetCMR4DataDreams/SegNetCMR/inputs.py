import tensorflow as tf

IMG_WIDTH = 256
IMG_HEIGHT = 256
# Number of color channels for the images: 1 channel for gray-scale.
NUM_COLOR_CHANNELS = 1

def placeholder_inputs(batch_size):

    images = tf.placeholder(tf.float32, [batch_size, IMG_WIDTH, IMG_HEIGHT, NUM_COLOR_CHANNELS])
    labels = tf.placeholder(tf.int64, [batch_size, IMG_WIDTH, IMG_HEIGHT])
    is_training = tf.placeholder(tf.bool)

    return images, labels, is_training
