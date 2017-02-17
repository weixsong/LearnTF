#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
learning_rate = 0.01
training_iters = 1000000
display_step = 100

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w):
    # notice we use the same model as linear regression, 
    # this is because there is a baked in cost function which performs softmax and cross entropy
    out = tf.add(tf.matmul(X, w), b)
    out = tf.nn.sigmoid(out)
    return out


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784]) # create symbolic variables
Y = tf.placeholder("float", [None, 10])

# like in linear regression, we need a shared variable weight matrix for logistic regression
w = init_weights([784, 10])
b = tf.Variable(tf.random_normal([10]))

py_x = model(X, w)

 # compute mean cross entropy (softmax is applied internally)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))

# construct optimizer
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(py_x, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# init
init = tf.global_variables_initializer()

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    sess.run(init)

    step = 0
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        if step % display_step == 0:
            acc, loss = sess.run([accuracy, cost], 
                feed_dict={X: mnist.test.images[:256], Y: mnist.test.labels[:256]})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1

    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                      Y: mnist.test.labels[:256],}))
