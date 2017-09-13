'''
Mixture Density Network is showed in this tutorail.
MDN (Mixture Density Network) is a deep nework but the output layer
will learn a distribution by GMM.

Author: Wei Song
'''


import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt

# generate fake training data
# build this model use fake data, input is x, target is y. the network will learn
# what is the the output y given input x
# y = 7 * sin(0.75x) + 0.5x + e
NSAMPLE = 1000
x_data = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T # NSAMPLE * 1
r_data = np.float32(np.random.normal(size=(NSAMPLE,1)))
y_data = np.float32(np.sin(0.75 * x_data) * 7.0 + x_data * 0.5 + r_data * 1.0)

plt.figure(figsize=(8, 8))
plot_out = plt.plot(x_data,y_data,'ro',alpha=0.3)
plt.show()


# create placeholder
x = tf.placeholder(dtype=tf.float32, shape=[None,1])
y = tf.placeholder(dtype=tf.float32, shape=[None,1])


# define a simple network with only one layer
NHIDDEN = 20
W = tf.Variable(tf.random_normal([1, NHIDDEN], stddev=1.0, dtype=tf.float32))
b = tf.Variable(tf.random_normal([1, NHIDDEN], stddev=1.0, dtype=tf.float32))

hidden_layer = tf.nn.tanh(tf.matmul(x, W) + b)

# create output layer, in output layer we need to predict y given input x
# output layer is a regression layer
W_out = tf.Variable(tf.random_normal([NHIDDEN,1], stddev=1.0, dtype=tf.float32))
b_out = tf.Variable(tf.random_normal([1,1], stddev=1.0, dtype=tf.float32))


y_out = tf.matmul(hidden_layer,W_out) + b_out


# define a loss
loss = tf.nn.l2_loss(y_out - y);

optimizer = tf.train.RMSPropOptimizer(learning_rate=0.1, decay=0.8).minimize(loss)


# Initializing the variables
init = tf.global_variables_initializer()

NEPOCH = 1000
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    for i in range(NEPOCH):
        sess.run(optimizer, feed_dict={x: x_data, y: y_data})
        
    x_test = np.float32(np.arange(-10.5, 10.5, 0.1))
    x_test = x_test.reshape(x_test.size, 1)
    y_test = sess.run(y_out, feed_dict={x: x_test})
    
    plt.figure(figsize=(8, 8))
    plt.plot(x_data, y_data, 'ro', x_test, y_test, 'bo', alpha=0.3)
    plt.show()

# now in the figure we see that our simple regression network learn the simple sinusoidal data very well.
# However, this kind of network only works when the function we want to learn by network is one-to-one function.


    
