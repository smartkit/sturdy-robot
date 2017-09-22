import cv2
import numpy as np
import tensorflow as tf

import os

import tensorflow as tf

import SegNetCMR as sn

HAVE_GPU = False
SAVE_INTERVAL = 2

TRAINING_DIR = './Data/Training'
TEST_DIR = './Data/Test'

RUN_NAME = "Run3x3"
CONV_SIZE = 3

ROOT_LOG_DIR = './Output'
CHECKPOINT_FN = 'model.ckpt'

#Start off at 0.9, then increase.
BATCH_NORM_DECAY_RATE = 0.9

MAX_STEPS = 10
BATCH_SIZE = 1

LOG_DIR = os.path.join(ROOT_LOG_DIR, RUN_NAME)
CHECKPOINT_FL = os.path.join(LOG_DIR, CHECKPOINT_FN)

def main():
    training_data = sn.GetData(TRAINING_DIR)
    test_data = sn.GetData(TEST_DIR)

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

        init = tf.global_variables_initializer()

        # saver = tf.train.Saver([x for x in tf.global_variables()])
        saver = tf.train.import_meta_graph(ROOT_LOG_DIR+"/"+RUN_NAME+"/"+CHECKPOINT_FN+"-10.meta")

        sm = tf.train.SessionManager()


        with sm.prepare_session("", init_op=init, saver=saver, checkpoint_dir=LOG_DIR) as sess:

            saver.restore(sess, tf.train.latest_checkpoint(ROOT_LOG_DIR+"/"+RUN_NAME))
            v_images=[]
            image = cv2.imread("./Data/Training/Images/normal/normal1.ndpi.16.5702_35104.2048x2048.png")
            # Resizing the image to our desired size and
            # preprocessing will be done exactly as done during training
            image = cv2.resize(image, (256, 256), cv2.INTER_LINEAR)
            v_images.append(image)
            v_images = np.array(v_images, dtype=np.uint8)
            v_images = v_images.astype('float32')
            v_images = np.multiply(v_images, 1.0/255.0)
            #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
            # x_batch = v_images.reshape(1, 256,256,1)
            x_batch = images[0]


            graph = tf.get_default_graph()

            y_pred = graph.get_tensor_by_name("unpool:0")

            ## Let's feed the images to the input placeholders
            x= graph.get_tensor_by_name("x:0")
            y_true = graph.get_tensor_by_name("y_true:0")
            y_test_images = np.zeros((1, 2))

            feed_dict_testing = {x: x_batch, y_true: y_test_images}
            result=sess.run(y_pred, feed_dict=feed_dict_testing)
            print('NN predicted:', result)

            # sess.run(tf.variables_initializer([x for x in tf.global_variables() if 'Adam' in x.name]))
            #
            # train_writer = tf.summary.FileWriter(LOG_DIR + "/Train", sess.graph)
            # test_writer = tf.summary.FileWriter(LOG_DIR + "/Test")
            #
            # global_step_value, = sess.run([global_step])
            #
            # print("Last trained iteration was: ", global_step_value)
            #
            # for step in range(global_step_value+1, global_step_value+MAX_STEPS+1):
            #
            #     print("Iteration: ", step)
            #
            #     images_batch, labels_batch = training_data.next_batch(BATCH_SIZE)
            #
            #     train_feed_dict = {images: images_batch,
            #                        labels: labels_batch,
            #                        is_training: True}
            #
            #     _, train_loss_value, train_accuracy_value, train_summary_str = sess.run([train_op, loss, accuracy, summary], feed_dict=train_feed_dict)
            #
            #     if step % SAVE_INTERVAL == 0:
            #
            #         print("Train Loss: ", train_loss_value)
            #         print("Train accuracy: ", train_accuracy_value)
            #         train_writer.add_summary(train_summary_str, step)
            #         train_writer.flush()
            #
            #         images_batch, labels_batch = test_data.next_batch(BATCH_SIZE)
            #
            #         test_feed_dict = {images: images_batch,
            #                           labels: labels_batch,
            #                           is_training: False}
            #
            #         test_loss_value, test_accuracy_value, test_summary_str = sess.run([loss, accuracy, summary], feed_dict=test_feed_dict)
            #
            #         print("Test Loss: ", test_loss_value)
            #         print("Test accuracy: ", test_accuracy_value)
            #         test_writer.add_summary(test_summary_str, step)
            #         test_writer.flush()
            #
            #         saver.save(sess, CHECKPOINT_FL, global_step=step)
            #         print("Session Saved")
            #         print("================")


if __name__ == '__main__':
    main()

