import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

A_r = np.load("A.npy")
# A_r = A_r[np.newaxis, :, :, np.newaxis]

W_from_Inception000 = np.load("val_000.npy")
W_from_Inception000_1 = W_from_Inception000[:, :, :, 0]
W_from_Inception000_1 = W_from_Inception000_1[:, :, :, np.newaxis]

xs = tf.Variable(tf.random_normal([1, 299, 299, 3], stddev=0.1))
conv_1 = tf.nn.conv2d(xs, W_from_Inception000_1, strides=[1, 1, 1, 1], padding='SAME')
Rule_1 = tf.nn.relu(conv_1)
pooling_1 = tf.nn.max_pool(Rule_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

Loss = tf.reduce_sum(tf.square(tf.subtract(conv_1, A_r)))
trainer = tf.train.GradientDescentOptimizer(0.001).minimize(Loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500):
    L, _ = sess.run([Loss, trainer])
    print("i: {}, Loss: {}".format(i, L))

cal_ori_img = sess.run(xs)
cal_ori_img = cal_ori_img[0]
plt.imshow(cal_ori_img)
plt.show()
