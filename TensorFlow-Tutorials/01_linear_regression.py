#!/usr/bin/env python

import tensorflow as tf
import numpy as np

trX = np.linspace(-1, 1, 101)
# create a y value which is approximately linear but with some random noise
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33

X = tf.placeholder("float") # create symbolic variables
Y = tf.placeholder("float")

def model(X, w):
    # lr is just X*w so this model line is pretty simple
    return tf.multiply(X, w)


# create a shared variable (like theano.shared) for the weight matrix
w = tf.Variable(0.0, name="weights")
y_model = model(X, w)

# use square error for cost function
cost = tf.square(Y - y_model)

# construct an optimizer to minimize cost and fit line to my data
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize variables (in this case just variable W)
    sess.run(init)

    for i in range(100):
        if i % 10 == 0:
            print("train step %d" % i)

        for (x, y) in zip(trX, trY):
            sess.run(train_op, feed_dict={X: x, Y: y})

    print(sess.run(w))  # It should be something around 2
