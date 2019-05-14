import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X_train = mnist.train.images
Y_train = mnist.train.labels
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

reshape_X = tf.reshape(xs, [-1, 28, 28, 1])


def FC(X, in_channel, out_channel, active_function=tf.nn.relu):
    W = tf.Variable(tf.random_normal([in_channel, out_channel], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[out_channel]))
    L = tf.matmul(X, W) + b
    if active_function != None:
        L = active_function(L)
    return L


def CNN(X, in_channel, out_channel, kernel_size=3):
    W_conv = tf.Variable(tf.random_normal([kernel_size, kernel_size, in_channel, out_channel], stddev=0.1))
    b_conv = tf.Variable(tf.constant(0.1, shape=[out_channel]))
    conv = tf.nn.relu(tf.nn.conv2d(X, W_conv, strides=[1, 1, 1, 1], padding='SAME') + b_conv)
    pooling = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 13*13
    return pooling


def save():
    saver = tf.train.Saver()
    saver.save(sess, "models/clean/clean.ckpt")


def restore():
    saver = tf.train.Saver()
    saver.restore(sess, "models/clean.ckpt")


cnn1 = CNN(reshape_X, 1, 8, 3)
cnn2 = CNN(cnn1, 8, 16, 3)
cnn3 = CNN(cnn2, 16, 16, 3)

flatten = tf.reshape(cnn3, [-1, 4 * 4 * 16])

f1 = FC(flatten, 4 * 4 * 16, 1024)
f2 = FC(f1, 1024, 1024)

# dropout
keep_prob = tf.placeholder("float")
f2_drop = tf.nn.dropout(f2, keep_prob)

y_out = FC(f2_drop, 1024, 10, None)

# 训练过程
loss = tf.losses.softmax_cross_entropy(ys, y_out)
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(loss)
correct_p = tf.equal(tf.argmax(y_out, 1), (tf.argmax(ys, 1)))
accuracy = tf.reduce_mean(tf.cast(correct_p, "float"))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("开始训练:")
for i in range(6000):
    batch = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={xs: batch[0], ys: batch[1], keep_prob: 0.5})
    if i % 600 == 0:
        print(i / 600)
        print(sess.run(accuracy, feed_dict={xs: batch[0], ys: batch[1], keep_prob: 1.0}))
print("进行测试集测试:")
testbatch = mnist.test.next_batch(1000)
print(sess.run(accuracy, feed_dict={xs: testbatch[0], ys: testbatch[1], keep_prob: 1.0}))
save()
