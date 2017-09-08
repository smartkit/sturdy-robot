import tensorflow as tf

def placeholder_inputs(batch_size):

    images = tf.placeholder(tf.float32, [batch_size, 2048, 2048, 1])
    labels = tf.placeholder(tf.int64, [batch_size, 2048, 2048])
    is_training = tf.placeholder(tf.bool)

    return images, labels, is_training