import tensorflow as tf


def loss_calc(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss', loss)
    return loss


def evaluation(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 3), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy

def prediction(sess,logits, labels):
    # num = randint(0, images.shape[0])
    # img = images[num]
    classification = sess.run(tf.argmax(logits, 3), feed_dict={images: [img]})
    # plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
    # plt.show()
    print('NN predicted:', classification)
    return classification
