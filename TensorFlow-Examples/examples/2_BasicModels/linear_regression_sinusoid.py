'''
A linear regression learning algorithm example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy
import numpy as np
import matplotlib.pyplot as plt
rng = numpy.random

# Parameters
learning_rate = 0.001
training_steps = 1000
display_step = 50
sample_number = 1000
linspace_start = 0
linespace_end = 4 * np.pi
batch_size = 20

# training data
train_x = numpy.linspace(linspace_start, linespace_end, num=sample_number)
train_y = numpy.sin(train_x)
train_y = np.reshape(train_y, [sample_number, 1])
n_samples = sample_number

# # draw train data
# print("close the image to continue training")
# plt.plot(train_x, train_y, 'b', label='Original sinusoid data')
# plt.ylabel('Y')
# plt.xlabel('X')
# plt.legend()
# plt.show()

# because we need to fit a sinusoid, only one parameter is not enouth
# so we need to increase parameter number, this need us to increase
# the feature number
# here we constuct a few other feature by train_x
origin_x = train_x
for i in range(2):
    temp = np.power(origin_x, i + 2)
    train_x = np.column_stack((train_x, temp))

# normalize data
train_x = (train_x - train_x.mean(0)) / train_x.std(0)

print("training data shape")
print(train_x.shape)

# tf Graph Input
X = tf.placeholder("float", [batch_size, 3])
Y = tf.placeholder("float", [batch_size, 1])

# Set model weights
W = tf.Variable(tf.random_normal([3, 1]), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.matmul(X, W), b)

# Mean squared error
cost = tf.reduce_mean(tf.reduce_sum(tf.pow(pred - Y, 2)) / 2)

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for step in range(training_steps):
        s = (step * batch_size) % sample_number
        e = (step * batch_size) % sample_number + batch_size

        x = train_x[s:e]
        y = train_y[s:e]
        sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (step + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: x, Y: y})
            print("step:", '%04d' % (step + 1), "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Graphic display
    plt.plot(origin_x, train_y, 'r', label='Original data')
    # plt.plot(train_x, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.legend()
    plt.show()
