import numpy as np
import tensorflow as tf

import os

import tensorflow as tf

import SegNetCMR as sn

HAVE_GPU = False

PREDICT_DIR = "./Data/Predict/201709/"

RUN_NAME = "Run3x3"
CONV_SIZE = 3

ROOT_LOG_DIR = './Output'
CHECKPOINT_FN = 'model.ckpt'

BATCH_SIZE = 1

LOG_DIR = os.path.join(ROOT_LOG_DIR, RUN_NAME)
CHECKPOINT_FL = os.path.join(LOG_DIR, CHECKPOINT_FN)

"""
Multi dimensional softmax,
refer to https://github.com/tensorflow/tensorflow/issues/210
compute softmax along the dimension of target
the native softmax only supports batch_size x dimension
"""
def softmax(target, axis, name=None):
    with tf.name_scope(name, 'softmax', values=[target]):
        max_axis = tf.reduce_max(target, axis, keep_dims=True)
        target_exp = tf.exp(target-max_axis)
        normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
        softmax = target_exp / normalize
        return softmax

def main():

    predict_data = sn.GetData(PREDICT_DIR, 'predict')

    g = tf.Graph()


    with g.as_default():

        images, labels, is_training = sn.placeholder_inputs(batch_size=BATCH_SIZE)

        # logits = sn.inference(images=images, is_training=is_training, conv_size=CONV_SIZE, batch_norm_decay_rate=BATCH_NORM_DECAY_RATE, have_gpu=HAVE_GPU)
        #
        # sn.add_output_images(images=images, logits=logits, labels=labels)
        #
        # loss = sn.loss_calc(logits=logits, labels=labels)
        #
        # train_op, global_step = sn.training(loss=loss, learning_rate=1e-04)
        #
        # accuracy = sn.evaluation(logits=logits, labels=labels)
        #
        # summary = tf.summary.merge_all()

        # init = tf.global_variables_initializer()

        sess = tf.Session()
        # First let's load meta graph and restore weights
        # saver = tf.train.Saver([x for x in tf.global_variables()])
        saver = tf.train.import_meta_graph(ROOT_LOG_DIR + "/" + RUN_NAME + "/" + CHECKPOINT_FN + "-22.meta")
        saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_FL))

        # Now, let's access and create placeholders variables and
        # create feed-dict to feed new data

        graph = tf.get_default_graph()
        w1 = graph.get_tensor_by_name("w1:0")
        # w2 = graph.get_tensor_by_name("w2:0")
        # feed_dict = {w1: 13.0, w2: 17.0}
        #
        # # Now, access the op that you want to run.
        # op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
        #
        # print(sess.run(op_to_restore, feed_dict))

        sm = tf.train.SessionManager()

if __name__ == '__main__':
    main()

