import PIL
import numpy as np
import tensorflow as tf

img_path = "cat.jpg"
img_class = 281
img = PIL.Image.open(img_path)
big_dim = max(img.width, img.height)
wide = img.width > img.height
new_w = 299 if not wide else int(img.width * 299 / img.height)
new_h = 299 if wide else int(img.height * 299 / img.width)
img = img.resize((new_w, new_h)).crop((0, 0, 299, 299))
img = (np.asarray(img) / 255.0).astype(np.float32)

W_from_Inception000 = np.load("val_000.npy")
W_from_Inception000_1 = W_from_Inception000[:, :, :, 0]
W_from_Inception000_1 = W_from_Inception000_1[:, :, :, np.newaxis]

xs = tf.placeholder(tf.float32, [1, 299, 299, 3])
W_conv1 = tf.Variable(W_from_Inception000_1)
conv_1 = tf.nn.conv2d(xs, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
Rule_1 = tf.nn.relu(conv_1)
pooling_1 = tf.nn.max_pool(Rule_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

sess = tf.Session()
sess.run(tf.global_variables_initializer())
C, R, P = sess.run([conv_1, Rule_1, pooling_1], feed_dict={xs: [img]})

np.save("A.npy", C)
np.save("R.npy", R)
np.save("P.npy", P)
