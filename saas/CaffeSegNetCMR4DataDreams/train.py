import os

import tensorflow as tf

import SegNetCMR as sn

# import matplotlib
# # matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

from random import randint

HAVE_GPU = False
SAVE_INTERVAL = 2

TRAINING_DIR = './Data/Train'
TEST_DIR = './Data/Test'
# VALID_DIR = './Data/Valid'
PREDICT_DIR = "./Data/Predict/201709/"

RUN_NAME = "Run3x3"
CONV_SIZE = 3

ROOT_LOG_DIR = './Output'
CHECKPOINT_FN = 'model.ckpt'

#Start off at 0.9, then increase.
BATCH_NORM_DECAY_RATE = 0.9

MAX_STEPS = 1000
BATCH_SIZE = 1

LOG_DIR = os.path.join(ROOT_LOG_DIR, RUN_NAME)
CHECKPOINT_FL = os.path.join(LOG_DIR, CHECKPOINT_FN)

test_accuracy_value = 0.0
PREDICT_RATE = 0.5


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

def prediction(sess,logits, SegnetCMR):
    # num = randint(0, images.shape[0])
    # img = images[num]
    classification = sess.run(tf.argmax(logits, 3), feed_dict={images: SegnetCMR.images})
    # plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
    # plt.show()
    print('NN predicted:', classification)
    return classification

def main():
    training_data = sn.GetData(TRAINING_DIR,'train')
    test_data = sn.GetData(TEST_DIR,'test')
    # val_data = sn.GetData(VALID_DIR)
    predict_data = sn.GetData(PREDICT_DIR,'predict')

    g = tf.Graph()

    with g.as_default():

        images, labels, is_training = sn.placeholder_inputs(batch_size=BATCH_SIZE)

        logits = sn.inference(images=images, is_training=is_training, conv_size=CONV_SIZE, batch_norm_decay_rate=BATCH_NORM_DECAY_RATE, have_gpu=HAVE_GPU)
        # print("logits:",logits);
        # In order to get probabilities we apply softmax on the output.
        # probabilities = tf.nn.softmax(logits)

        # For each pixel we get predictions for each class
        # out of 1000. We need to pick the one with the highest
        # probability. To be more precise, these are not probabilities,
        # because we didn't apply softmax. But if we pick a class
        # with the highest value it will be equivalent to picking
        # the highest value after applying softmax
        probabilities = tf.argmax(logits, dimension=3)
        print("probabilities:", probabilities)
        sn.add_output_images(images=images, logits=logits, labels=labels)

        loss = sn.loss_calc(logits=logits, labels=labels)

        #TODO: Optimizer.
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
        #TODO: drop-out during training

        # Predictions for the training, validation, and test data.
        train_prediction = tf.equal(tf.argmax(logits, 3), labels)
        print("train_prediction:",train_prediction)
        # valid_prediction = tf.nn.softmax(model(tf_valid_dataset, False))
        # test_prediction = tf.nn.softmax(model(tf_test_dataset, False))

        train_op, global_step = sn.training(loss=loss, learning_rate=1e-04)

        accuracy = sn.evaluation(logits=logits, labels=labels)

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver([x for x in tf.global_variables() if 'Adam' not in x.name])
        # saver = tf.train.Saver([x for x in tf.global_variables()])

        sm = tf.train.SessionManager()



        with sm.prepare_session("", init_op=init, saver=saver, checkpoint_dir=LOG_DIR) as sess:

            sess.run(tf.variables_initializer([x for x in tf.global_variables() if 'Adam' in x.name]))
            # sess.run(tf.variables_initializer([x for x in tf.global_variables()]))

            train_writer = tf.summary.FileWriter(LOG_DIR + "/Train", sess.graph)
            test_writer = tf.summary.FileWriter(LOG_DIR + "/Test")

            global_step_value, = sess.run([global_step])

            print("Last trained iteration was: ", global_step_value)

            for step in range(global_step_value+1, global_step_value+MAX_STEPS+1):

                print("Iteration: ", step)

                images_batch, labels_batch, fnames_batch = training_data.next_batch(BATCH_SIZE)

                train_feed_dict = {images: images_batch,
                                   labels: labels_batch,
                                   is_training: True}

                _, train_loss_value, train_accuracy_value, train_summary_str = sess.run([train_op, loss, accuracy, summary], feed_dict=train_feed_dict)

                global test_accuracy_value

                if step % SAVE_INTERVAL == 0:

                    print("Train Loss: ", train_loss_value)
                    print("Train accuracy: ", train_accuracy_value)
                    train_writer.add_summary(train_summary_str, step)
                    train_writer.flush()

                    images_batch, labels_batch, fnames_batch = test_data.next_batch(BATCH_SIZE)

                    test_feed_dict = {images: images_batch,
                                      labels: labels_batch,
                                      is_training: False}

                    test_loss_value, test_accuracy_value, test_summary_str = sess.run([loss, accuracy, summary], feed_dict=test_feed_dict)

                    print("Test Loss: ", test_loss_value)
                    print("Test accuracy: ", test_accuracy_value)
                    test_writer.add_summary(test_summary_str, step)
                    test_writer.flush()

                    saver.save(sess, CHECKPOINT_FL, global_step=step)

                    print("Session Saved")
                    print("================")

                if test_accuracy_value > PREDICT_RATE:

                    images_batch, labels_batch, fnames_batch = predict_data.next_batch(1)

                    predict_feed_dict = {images: images_batch,
                                      labels: labels_batch,
                                      is_training: False}

                    predict_loss_value, predict_accuracy_value, predict_summary_str = sess.run([loss, accuracy, summary],
                                                                                      feed_dict=predict_feed_dict)
                    print("Predict Loss: ", predict_loss_value)
                    print("Predict accuracy: ", predict_accuracy_value)

                    dense_prediction, im = sess.run([logits, probabilities], feed_dict=predict_feed_dict)

                    # print("dense_prediction:",dense_prediction)
                    # print("im:", im)

                    pred = tf.argmax(logits, dimension=-1)
                    # pred = tf.nn.softmax(logits=logits, dim=-1)[..., 1]

                    # Make a queue of file names including all the JPEG images files in the relative
                    # image directory.
                    filename_queue = tf.train.string_input_producer(
                        tf.train.match_filenames_once(fnames_batch))

                    # Read an entire image file which is required since they're JPEGs, if the images
                    # are too large they could be split in advance to smaller files or use the Fixed
                    # reader to split up the file.
                    image_reader = tf.WholeFileReader()

                    # Read a whole file from the queue, the first returned value in the tuple is the
                    # filename which we are ignoring.
                    _, image_file = image_reader.read(filename_queue)

                    # Decode the image as a JPEG file, this will turn it into a Tensor which we can
                    # then use in training.
                    pred_image = tf.image.decode_png(image_file, channels=3)

                    # Write file result
                    text_file = open("smartkit.ai.txt", "w")
                    text_file.write(fnames_batch[0])
                    text_file.write("\t")
                    text_file.write(" F")
                    text_file.close()

                    segmentation, np_image = sess.run([pred, pred_image])


                    # predicted = prediction(sess,logits,predict_data)
                    # classification = sess.run(tf.argmax(logits, 3), feed_dict={images: predict_data.images[0]})
                    classification = sess.run(tf.nn.softmax(logits,3,name="predict"), feed_dict={images: predict_data.images[0]})
                    # plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
                    # plt.show()
                    print('NN predicted:', classification)
                    layer_fc2 = tf.Graph.get_tensor_by_name("stride/slice:0")
                    y_pred = tf.nn.softmax(layer_fc2, name="y_pred")
                    y_pred_cls = tf.argmax(y_pred, dimension=1)
                    return classification

#@see: https://github.com/burliEnterprises/tensorflow-image-classifier
                    # softmax_tensor = sess.graph.get_tensor_by_name('pool:5')
                    # return: Tensor("final_result:0", shape=(?, 4), dtype=float32); stringname definiert in retrain.py, zeile 1064

                    # predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
                    # gibt prediction values in array zuerueck:

                    # top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                    # sortierung; circle -> 0, plus -> 1, square -> 2, triangle -> 3; array return bsp [3 1 2 0] -> sortiert nach groesster uebereinstimmmung
#@see: https://stackoverflow.com/questions/42211833/tensorflow-how-to-predict-with-trained-model-on-a-different-test-dataset
                    # ckpt = tf.train.get_checkpoint_state('./model/')
                    # saver.restore(sess, ckpt.model_checkpoint_path)
                    # feed_dict = {training_data: images_batch}
                    # predictions = sess.run([test_prediction], feed_dict)
#metrics,@see: https://stackoverflow.com/questions/35365007/tensorflow-precision-recall-f1-score-and-confusion-matrix
                # @see: https://gist.github.com/Mistobaan/337222ac3acbfc00bdac
                    predicted = tf.round(tf.nn.sigmoid(logits))
                    # predicted = tf.cast(tf.equal(tf.argmax(logits, 3), labels),tf.int64)
                    actual = labels

                    # Count true positives, true negatives, false positives and false negatives.
                    tp = tf.count_nonzero(predicted * actual)
                    tn = tf.count_nonzero((predicted - 1) * (actual - 1))
                    fp = tf.count_nonzero(predicted * (actual - 1))
                    fn = tf.count_nonzero((predicted - 1) * actual)
                    print("tp,tn,fp,fn:",tp,tn,fp,fn)

                    # Calculate accuracy, precision, recall and F1 score.
                    m_accuracy = (tp + tn) / (tp + fp + fn + tn)
                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    fmeasure = (2 * precision * recall) / (precision + recall)
                    print("accuracy,precision,recall,fmeature:", m_accuracy, precision, recall, fmeasure)

                    # Add metrics to TensorBoard.
                    tf.summary.scalar('Accuracy', m_accuracy)
                    tf.summary.scalar('Precision', precision)
                    tf.summary.scalar('Recall', recall)
                    tf.summary.scalar('f-measure', fmeasure)


if __name__ == '__main__':
    main()
