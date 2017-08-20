# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import logging

import tensorflow as tf
import  numpy as np

#define some constant
NUMBER_FEATURES = 1      #number input features
LEARNING_RATE = 10e-3    #learning rate
NUMBER_EPOCH = 3         #number epoch
BATCH_SIZE = 2           #batch size with each step training

# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
# Visualize data

def mat_mul(inputs):
    with tf.variable_scope("mat_mul"):
        W = tf.get_variable(name= "weight", shape=
                            [NUMBER_FEATURES, 1], initializer=tf.random_normal_initializer())
        bias = tf.get_variable(name= "bias", shape= [1],
                               initializer=tf.constant_initializer(0.1))
        mat_mul = tf.matmul(inputs, W)+ bias
    return mat_mul


def loss(mat_mul, labels):
    with tf.variable_scope("loss"):
        osl = tf.losses.mean_squared_error(labels= labels, predictions= mat_mul)
    return osl

def train():
    inputs = tf.placeholder(dtype= tf.float32, shape=[BATCH_SIZE, NUMBER_FEATURES])
    labels = tf.placeholder(dtype= tf.float32, shape=[BATCH_SIZE,1])

    opt = tf.train.GradientDescentOptimizer(learning_rate=1e-6)
    with tf.variable_scope("training"):
        _mat_mul = mat_mul(inputs)
        _loss = loss(_mat_mul, labels)
        _train = opt.minimize(_loss)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    logging.basicConfig(level=logging.INFO)
    for j in range(20):
        for i in range(5):

            in_ = X[i*2 : (i*2+ 2), :]
            lab_ = y[i * 2: (i * 2 + 2), :]
            feed = {inputs:in_, labels:lab_}
            mat_, loss_, _ = sess.run([_mat_mul, _loss, _train], feed_dict= feed)
            logging.info("loss: " +str(loss_))


if __name__ == '__main__':
    train()