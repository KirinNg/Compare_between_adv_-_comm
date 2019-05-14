import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import numpy as np
import os
import PIL
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

_BATCH_SIZE = 1
X = tf.placeholder(tf.float32, [_BATCH_SIZE, 299, 299, 3])
Y = tf.placeholder(tf.int32, [_BATCH_SIZE])
LABEL = tf.placeholder(tf.int32)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

def inception(image, reuse=tf.AUTO_REUSE):
    preprocessed = tf.multiply(tf.subtract(image, 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, end_point = nets.inception.inception_v3(preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:, 1:]  # ignore background class
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point

def load_img(path):
    I = PIL.Image.open(path).convert('RGB')
    I = I.resize((299, 299)).crop((0, 0, 299, 299))
    I = (np.asarray(I) / 255.0).astype(np.float32)
    return I[:, :, 0:3]

logits, probs, end_point = inception(X)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, "inception/models/inception_v3_2016_08_28/inception_v3.ckpt")

val = tf.global_variables()

v000 = sess.run(val[0])
np.save("inception/val_000.npy", v000)
