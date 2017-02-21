"""
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Wei Song
Project: https://github.com/weixsong/LearnTF
"""

from __future__ import print_function

import tensorflow as tf
import threading

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

# Parameters
GPU_NUMS = 2
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784   # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


class CustomRunner(object):
    """
    This class manages the background threads needed to fill a queue full of data
    """

    def __init__(self, coord):
        self.coord = coord
        self.threads = []
        self.x_placeholder = tf.placeholder(tf.float32, [None, n_input])
        self.y_placeholder = tf.placeholder(tf.float32, [None, n_classes])

        self.queue = tf.PaddingFIFOQueue(32,
                                         shapes=[[None, n_input], [None, n_classes]],
                                         dtypes=[tf.float32, tf.float32])
        # enqueue operation
        self.enqueue_op = self.queue.enqueue([self.x_placeholder, self.y_placeholder])

    def dequeue(self):
        x, y = self.queue.dequeue()
        return x, y

    def get_test_data(self):
        x = mnist.test.images[:256]
        y = mnist.test.labels[:256]
        return x, y

    def thread_main(self, sess):
        stop = False
        while not stop:
            if self.coord.should_stop():
                stop = True
                break

            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(self.enqueue_op, feed_dict={self.x_placeholder: batch_x,
                                                 self.y_placeholder: batch_y})

    def start_threads(self, sess, n_threads=1):
        """
        start background threads to feed queue
        """
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            # force the thread close when parent thread quit
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

        return self.threads


def average_gradients(tower_grads):
    """
    Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), (grad0_gpu1, var0_gpu1)... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            if g is None:
                continue

            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        if len(grads) == 0:
            average_grads.append((None, grad_and_vars[0][1]))
            continue

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, get_variable("wc1", [5, 5, 1, 32]), get_bias("bc1", [32]))
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, get_variable("wc2", [5, 5, 32, 64]), get_bias("bc1", [64]))
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    wd1 = get_variable("wd1", [7*7*64, 1024])
    bc1 = get_bias("bc1", [1024])
    fc1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, wd1), bc1)
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, get_variable("out", [1024, n_classes])), get_bias("bc1", [n_classes]))
    return out


def create_network(x, y, keep_prob):
    # Construct model
    pred = conv_net(x, keep_prob)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return loss, accuracy


def get_variable(name, shape):
    """create variable on CPU"""
    with tf.device("/cpu:0"):
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        variable = tf.get_variable(name, initializer(shape=shape))
    return variable


def get_bias(name, shape):
    """create variable on CPU"""
    with tf.device("/cpu:0"):
        initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
        variable = tf.get_variable(name, initializer(shape=shape))
    return variable


with tf.device("/cpu:0"):
    # Create coordinator.
    coord = tf.train.Coordinator()
    reader = CustomRunner(coord)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0), trainable=False)

    tower_grads = []
    tower_losses = []
    tower_accuracies = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(GPU_NUMS):
            with tf.device("/gpu:%d" % i), tf.name_scope("tower_%d" % i) as scope:
                # get data from reader
                x, y = reader.dequeue()

                # create network
                loss, accuracy = create_network(x, y, keep_prob)

                # compute gradients
                trainable = tf.trainable_variables()
                grads = optimizer.compute_gradients(loss, var_list=trainable)

                # append loss, grads, accuracy to towers
                tower_losses.append(loss)
                tower_grads.append(grads)
                tower_accuracies.append(accuracy)

                # reuse variable to create next network in next GPU
                tf.get_variable_scope().reuse_variables()

    # calculate the mean of each gradient. Synchronization point across all towers
    grads = average_gradients(tower_grads)
    train_ops = optimizer.apply_gradients(grads, global_step=global_step)

    # calculate the mean loss
    loss = tf.reduce_mean(tower_losses)

    # calculate the mean accuracy
    accuracy = tf.reduce_mean(tower_accuracies)

    # setup session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
    # Initializing the variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # start queue runner
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess=sess)

    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x2, batch_y2 = mnist.train.next_batch(batch_size)
        # Run optimization op
        sess.run(train_ops, feed_dict={keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss, accuracy], feed_dict={keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1

    print("Optimization Finished!")

    # Saver for storing checkpoints of the model
    saver = tf.train.Saver(var_list=tf.trainable_variables())

    # save model to disk
    model_path = "./model"
    save_path = saver.save(sess, model_path)
    print("Model saved in file %s" % save_path)

    # stop queue threads and close the session
    coord.request_stop()
    coord.join(threads)
    sess.close()

    # start a new session
    with tf.Session() as sess:
        sess.run(init)

        # restore model
        saver.restore(model_path)
        print("Model restored from file: %s" % save_path)

        # create network by model data
        x_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, n_input])
        y_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])
        loss, accuracy = create_network(x_placeholder, y_placeholder, keep_prob)

        # run test accuracy
        batch_x, batch_y = reader.get_test_data()
        accuracy_val = sess.run(accuracy, feed_dict={x_placeholder: batch_x,
                                                     y_placeholder: batch_y,
                                                     keep_prob: 1.0})

        print("Test Accuracy is %f" % accuracy_val)
