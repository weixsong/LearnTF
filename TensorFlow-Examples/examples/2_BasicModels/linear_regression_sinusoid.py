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
learning_rate = 0.01
training_steps = 10000
display_step = 50
sample_number = 1000
linspace_start = 0
linespace_end = 4 * np.pi
batch_size = 20

# training data
train_x = numpy.linspace(linspace_start, linespace_end, num=sample_number)
train_y = numpy.sin(train_x)
n_samples = sample_number

# draw train data
plt.plot(train_x, train_y, 'b', label='Original data')
plt.ylabel('Y')
plt.xlabel('X')
plt.legend()
plt.show()

# because we need to fit a sinusoid, only one parameter is not enouth
# so we need to increase parameter number, this need us to increase
# the feature number
# here we constuct a few other feature by train_x
for i in range(2):
    temp = np.power(train_x, i + 2)
    train_x = np.column_stack((train_x, temp))

print("training data shape")
print(train_x.shape)

# tf Graph Input
X = tf.placeholder("float", [batch_size, 3])
Y = tf.placeholder("float", [batch_size, 1])

# Set model weights
W = tf.Variable(tf.random_normal([10, 1]), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.reduce_sum(tf.multiply(X, W)), b)

# Mean squared error
cost = tf.reduce_mean(tf.pow(pred - Y, 2) / 2)

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_steps):
        for (x, y) in zip(train_x, train_y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_x, Y: train_y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c),
                  "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=",
          sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.legend()
    plt.show()
