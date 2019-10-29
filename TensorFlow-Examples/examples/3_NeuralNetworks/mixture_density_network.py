'''
Mixture Density Network is showed in this tutorial.
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
plt.title('y = 7 * sin(0.75x) + 0.5x + e')
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
loss = tf.nn.l2_loss(y_out - y)

optimizer = tf.train.RMSPropOptimizer(learning_rate=0.1, decay=0.8).minimize(loss)


# Initializing the variables
init = tf.global_variables_initializer()

NEPOCH = 1000
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    for i in range(NEPOCH):
        l, _ = sess.run([loss, optimizer], feed_dict={x: x_data, y: y_data})
        if i % 10 == 0:
            print("loss is: %s" % (str(l),))
        
    x_test = np.float32(np.arange(-10.5, 10.5, 0.1))
    x_test = x_test.reshape(x_test.size, 1)
    y_test = sess.run(y_out, feed_dict={x: x_test})
    
    plt.figure(figsize=(8, 8))
    plt.plot(x_data, y_data, 'ro', x_test, y_test, 'bo', alpha=0.3)
    plt.show()

# now in the figure we see that our simple regression network learn the simple sinusoidal data very well.
# However, this kind of network only works when the function we want to learn by network is one-to-one function.

# x = 7 * sin(0.75y) + 0.5y + e
temp_data = x_data
x_data = y_data
y_data = temp_data

plt.figure(figsize=(8, 8))
plot_out = plt.plot(x_data,y_data, 'ro', alpha=0.3)
plt.title("x = 7 * sin(0.75y) + 0.5y + e")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    for i in range(NEPOCH):
        l, _ = sess.run([loss, optimizer], feed_dict={x: x_data, y: y_data})
        if i % 10 == 0:
            print("loss is: %s" % (str(l),))
        
    x_test = np.float32(np.arange(-10.5, 10.5, 0.1))
    x_test = x_test.reshape(x_test.size, 1)
    y_test = sess.run(y_out, feed_dict={x: x_test})
    
    plt.figure(figsize=(8, 8))
    plt.plot(x_data,y_data,'ro', x_test,y_test, 'bo', alpha=0.3)
    plt.title("this kind of data learned very BAD!")
    plt.show()

# Our current model only predicts one output value for each input, 
# so this approach will fail miserably. What we want is a model that has the capacity 
# to predict a range of different output values for each input. 
# In the next section we implement a Mixture Density Network (MDN) to do achieve this task.

tf.reset_default_graph()

NHIDDEN = 24
STDEV = 0.5
KMIX = 24 # number of mixtures
NOUT = KMIX * 3 # pi, mu, stdev

x = tf.placeholder(dtype=tf.float32, shape=[None,1], name="x")
y = tf.placeholder(dtype=tf.float32, shape=[None,1], name="y")

Wh = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=STDEV, dtype=tf.float32))
bh = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=STDEV, dtype=tf.float32))

Wo = tf.Variable(tf.random_normal([NHIDDEN, NOUT], stddev=STDEV, dtype=tf.float32))
bo = tf.Variable(tf.random_normal([1, NOUT], stddev=STDEV, dtype=tf.float32))

hidden_layer = tf.nn.tanh(tf.matmul(x, Wh) + bh)
output = tf.matmul(hidden_layer, Wo) + bo


def get_mixture_coef(output):
    out_pi = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")
    out_sigma = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")
    out_mu = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")

    out_pi, out_sigma, out_mu = tf.split(output, 3, 1)

    max_pi = tf.reduce_max(out_pi, 1, keep_dims=True)
    out_pi = tf.subtract(out_pi, max_pi)

    out_pi = tf.exp(out_pi)

    normalize_pi = tf.reciprocal(tf.reduce_sum(out_pi, 1, keep_dims=True))
    out_pi = tf.multiply(normalize_pi, out_pi)

    out_sigma = tf.exp(out_sigma)

    return out_pi, out_sigma, out_mu


out_pi, out_sigma, out_mu = get_mixture_coef(output)
print(out_pi.shape)
print(out_sigma.shape)
print(out_mu.shape)

NSAMPLE = 2500
y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
r_data = np.float32(np.random.normal(size=(NSAMPLE, 1))) # random noise
x_data = np.float32(np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0)

plt.figure(figsize=(8, 8))
plt.plot(x_data, y_data, 'ro', alpha=0.3)
plt.show()


oneDivSqrtTwoPI = 1 / math.sqrt(2 * math.pi) # normalisation factor for gaussian, not needed.

# compute the probability of y given mu & sigma
def tf_normal(y, mu, sigma):
    print(y.shape)
    result = tf.subtract(y, mu)
    print(result.shape)
    result = tf.multiply(result, tf.reciprocal(sigma))
    result = -tf.square(result) / 2
    return tf.multiply(tf.exp(result), tf.reciprocal(sigma)) * oneDivSqrtTwoPI


def get_lossfunc(out_pi, out_sigma, out_mu, y):
    result = tf_normal(y, out_mu, out_sigma)
    result = tf.multiply(result, out_pi)
    result = tf.reduce_sum(result, 1, keep_dims=True)
    result = -tf.log(result)
    return tf.reduce_mean(result)

loss = get_lossfunc(out_pi, out_sigma, out_mu, y)
optimizer = tf.train.AdamOptimizer().minimize(loss)


# generate test data
x_test = np.float32(np.arange(-15, 15, 0.1))
NTEST = x_test.size
x_test = x_test.reshape(NTEST, 1) # needs to be a matrix, not a vector
# To sample a mixed gaussian distribution, we randomly select which distribution based on the set of  \Pi_{k} probabilities, 
# and then proceed to draw the point based off the  k^{th}gaussian distribution.

# get the Pi index given a random input x, by compute the accumulated pdf
# this Pi index is used to decide which Gaussian distribution is to use to sample output
def get_pi_idx(x, pdf):
    N = pdf.size
    accumulate = 0
    for i in range(0, N):
        accumulate += pdf[i]
        if accumulate >= x:
            return i
    print('error with sampling ensemble')
    return -1


# generate samples from the network learned distribution
def generate_ensemble(out_pi, out_mu, out_sigma, M=10):
    NTEST = x_test.size
    result = np.random.rand(NTEST, M) # initially random [0, 1]
    rn = np.random.randn(NTEST, M) # normal random matrix
    mu = 0
    std = 0
    idx = 0

    # transforms result into random ensembles
    for j in range(0, M):
        for i in range(0, NTEST):
            idx = get_pi_idx(result[i, j], out_pi[i])
            mu = out_mu[i, idx]
            std = out_sigma[i, idx]
            result[i, j] = mu + rn[i, j] * std

    return result

# let's train the model to see what will happen

NEPOCH = 10000

# Initializing the variables
init = tf.global_variables_initializer()

losses = [] # store the training progress here.
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    for i in range(NEPOCH):
        l, _ = sess.run([loss, optimizer], feed_dict={x: x_data, y: y_data})
        losses.append(l)
        if i % 10 == 0:
            print("MDN loss: %s, step is %s" % (str(l), str(i)))


    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(len(losses)), losses, 'r-')
    plt.title("loss")
    plt.show()

    # sample from GMM
    out_pi_test, out_sigma_test, out_mu_test = sess.run(get_mixture_coef(output), feed_dict={x: x_test})
    y_test = generate_ensemble(out_pi_test, out_mu_test, out_sigma_test)
    plt.figure(figsize=(8, 8))
    plt.plot(x_data, y_data, 'ro', x_test, y_test, 'bo', alpha=0.3)
    plt.show()

    # let's draw the u(x) distribution of all the Gaussian components
    print(x_test.shape)
    print(out_mu_test.shape)
    print(y_test.shape)
    plt.figure(figsize=(8, 8))
    plt.plot(x_test, out_mu_test, 'go', x_test, y_test, 'bo', alpha=0.3)
    plt.show()


    # let's draw the entire mixture pdf at each point on the x-axis to get a heatmap.

    x_heatmap_label = np.float32(np.arange(-15, 15, 0.1))
    y_heatmap_label = np.float32(np.arange(-15, 15, 0.1))

    # compute the probability of x given mu & std
    def custom_gaussian(x, mu, std):
        x_norm = (x - mu) / std
        result = oneDivSqrtTwoPI * math.exp(-x_norm*x_norm/2) / std
        return result

    def generate_heatmap(out_pi, out_mu, out_sigma, x_heatmap_label, y_heatmap_label):
        N = x_heatmap_label.size
        M = y_heatmap_label.size
        K = KMIX

        z = np.zeros((N, M)) # zero heat map
        mu, std, pi = 0, 0, 0

        # transforms result into random ensembles
        for k in range(0, K):
            for i in range(0, M):
                pi = out_pi[i, k]
                mu = out_mu[i, k]
                std = out_sigma[i, k]
                for j in range(0, N):
                    z[N-j-1, i] += pi * custom_gaussian(y_heatmap_label[j], mu, std)

        return z


    def draw_heatmap(xedges, yedges, heatmap):
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.figure(figsize=(8, 8))
        plt.imshow(heatmap, extent=extent)
        plt.show()
        
    z = generate_heatmap(out_pi_test, out_mu_test, out_sigma_test, x_heatmap_label, y_heatmap_label)
    draw_heatmap(x_heatmap_label, y_heatmap_label, z)
    
