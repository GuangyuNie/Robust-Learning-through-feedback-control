import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
# set random seed for comparing the two result calculations
#tf.set_random_seed(1)

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001
training_iters = 1000
batch_size = 128

n_inputs = 256   # LeNet output
n_steps = 24    # time steps
n_hidden_units = 32   # neurons in hidden layer
n_classes = 10      # MNIST classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, (None, 28, 28, 1))
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # (256, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Input = 28x28x1. Output = 24x24x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma,name='conv1_w'))
    conv1_b = tf.Variable(tf.zeros(6),name='conv1_b')
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 24x24x6. Output = 12x12x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 8x8x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma,name='conv2_w'))
    conv2_b = tf.Variable(tf.zeros(16),name='conv2_b')
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 8x8x16. Output = 4x4x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 4x4x16. Output = 256.
    fc0 = tf.contrib.layers.flatten(conv2)

    return fc0




def RNN(X, weights, biases):
    print(X.shape)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.ones([n_steps, tf.shape(X_in)[0], 1]) * X_in
    #X_in = tf.reshape(X_in, [batch_size, n_steps,None])
    print(X_in.shape)

    cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden_units,name = 'rnn_cell')
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=True)
    print(outputs.shape)
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (32, 10)
    print(results.shape)
    return results


latent = LeNet(x)
pred = RNN(latent, weights, biases)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(loss)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0

    while step < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, 28, 28, 1])
        batch_xs_test, batch_ys_test = mnist.test.next_batch(batch_size)
        batch_xs_test = batch_xs_test.reshape([batch_size, 28, 28, 1])
        _,total_loss = sess.run([train_op,loss], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            train_acc = sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
            })
        if step % 20 == 0:
            test_acc = sess.run(accuracy, feed_dict={
            x: batch_xs_test,
            y: batch_ys_test,
            })
            print('Train accuracy = {:.4f}, loss = {:.4f}, Test accuracy = {:.4f}'.format(train_acc,total_loss,test_acc))
        step += 1