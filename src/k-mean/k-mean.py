#Some comments and copy right placed here

"""
Using tutorial and some description about this source code
"""

from __future__ import print_function, division, absolute_import

from datetime import datetime
import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from classifier import Classifier

#Define some constants using in this model

########
np.random.seed(11)
class K_mean(Classifier):
    def __init__(self, data):
        self.data = data
        self.sess = tf.Session()
        Classifier.__init__(self)
        with tf.variable_scope("k_means"):
            rand_mean = np.random.choice(self.data.N*3, 3, replace=False)
            self.X = tf.placeholder(dtype=tf.float32,name="data", shape= [self.data.N*3,2])
            self.mean0 = tf.get_variable(name= "mean0", initializer= tf.constant(data.X[rand_mean[0]], dtype=tf.float32, shape=[1,2]), dtype=tf.float32)
            self.mean1 = tf.get_variable(name= "mean1", initializer= tf.constant(data.X[rand_mean[1]], dtype=tf.float32, shape=[1,2]), dtype=tf.float32)
            self.mean2 = tf.get_variable(name= "mean2", initializer= tf.constant(data.X[rand_mean[2]], dtype=tf.float32, shape=[1,2]), dtype=tf.float32)
            self.label = tf.get_variable(name= "label", initializer= tf.constant(np.zeros([self.data.N*3,1]), dtype=tf.float32), dtype=tf.float32)
            self.new_mean0 = tf.get_variable(name= "new_mean0", initializer= tf.constant(data.X[rand_mean[0]], dtype=tf.float32, shape=[1,2]), dtype=tf.float32)
            self.new_mean1 = tf.get_variable(name= "new_mean1", initializer= tf.constant(data.X[rand_mean[1]], dtype=tf.float32, shape=[1,2]), dtype=tf.float32)
            self.new_mean2 = tf.get_variable(name= "new_mean2", initializer= tf.constant(data.X[rand_mean[2]], dtype=tf.float32, shape=[1,2]), dtype=tf.float32)
        self.init = tf.global_variables_initializer()
    def get_name(self):
        return "K-means"
    def fit(self):
        #loop until the converged condition is got

        #init center
        #calculate data label
        # center
        new_label, re_ass_label = self.reassign_label()
        ass_new_mean, new_mean = self.update_center(new_label)
        converged, re_ass_mean = self._check_converge(new_mean)
        return  [new_label, re_ass_label], ass_new_mean, new_mean , converged, re_ass_mean
    def predict(self):
        #predict label for new data
        pass
    def _check_converge(self,new_mean):
        #check the algorithm is converged or not
        bool0 = tf.reduce_all(tf.equal(new_mean[0], self.mean0))
        bool1 = tf.reduce_all(tf.equal(new_mean[1], self.mean1))
        bool2 = tf.reduce_all(tf.equal(new_mean[2], self.mean2))
        re_ass_mean0 = tf.assign(self.mean0, new_mean[0])
        re_ass_mean1 = tf.assign(self.mean1, new_mean[1])
        re_ass_mean2 = tf.assign(self.mean2, new_mean[2])
        return tf.reduce_all([bool0, bool1, bool2]), [re_ass_mean0, re_ass_mean1, re_ass_mean2]
    def reassign_label(self):
        #reassign label for each data
        #calculate distance from this data to each of new center
        distance0 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.X, self.new_mean0)), axis=1))
        distance0 = tf.reshape(distance0, shape=[self.data.N*3,1])
        distance1 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.X, self.new_mean1)), axis=1))
        distance1 = tf.reshape(distance1, shape=[self.data.N*3,1])
        distance2 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.X, self.new_mean2)), axis=1))
        distance2 = tf.reshape(distance2, shape=[self.data.N*3,1])
        distance = tf.concat([distance0, distance1, distance2], axis=1)
        #choice center mean and assign new label for this data
        new_label = tf.cast(tf.arg_min(distance, dimension=1), dtype= tf.float32)
        new_label = tf.reshape(new_label, shape=[self.data.N*3,1])
        re_ass_label = tf.assign(self.label, new_label)
        return new_label, re_ass_label

    def update_center(self, new_label):
        #update for new center
        mask0 = tf.to_float(tf.equal(new_label, 0))
        number0 = tf.reduce_sum(mask0)
        mask1 = tf.to_float(tf.equal(new_label, 1))
        number1 = tf.reduce_sum(mask1)
        mask2 = tf.to_float(tf.equal(new_label, 2))
        number2 = tf.reduce_sum(mask2)
        new_mean0 = tf.reduce_sum(tf.multiply(self.X, mask0), axis=0)
        new_mean0 = tf.div(new_mean0, number0)
        new_mean0 = tf.reshape(new_mean0, shape=[1,2])
        new_mean1 = tf.reduce_sum(tf.multiply(self.X, mask1), axis=0)
        new_mean1 = tf.div(new_mean1, number1)
        new_mean1 = tf.reshape(new_mean1, shape=[1,2])
        new_mean2 = tf.reduce_sum(tf.multiply(self.X, mask2), axis=0)
        new_mean2 = tf.div(new_mean2, number2)
        new_mean2 = tf.reshape(new_mean2, shape=[1,2])
        assign0 = tf.assign(self.new_mean0, new_mean0)
        assign1 = tf.assign(self.new_mean1, new_mean1)
        assign2 = tf.assign(self.new_mean2, new_mean2)
        return [assign0, assign1, assign2], [new_mean0, new_mean1, new_mean2]
    def training(self):
        label, ass_new_mean, new_mean , converged, re_ass_mean = self.fit()
        self.sess.run(self.init)
        count = 0
        while True:
            _label, _ass_new_mean, _new_mean, _converged, _re_ass_mean = self.sess.run([label, ass_new_mean, new_mean, converged, re_ass_mean],feed_dict={self.X:self.data.X})
            print (_new_mean)
            if count == 0:
                count +=1
                print ("continue")
                continue
            else:
                pass
            if _converged:
                print (_converged)
                print ("Finish training")
                print ("Center0", self.sess.run(self.mean0))
                print ("Center1", self.sess.run(self.mean1))
                print ("Center2", self.sess.run(self.mean2))
                break
            else:
                print ("Center0", self.sess.run(self.mean0))
                print ("Center1", self.sess.run(self.mean1))
                print ("Center2", self.sess.run(self.mean2))

class Data:
    def __init__(self, means, conv, N):
        """

        :param means:
        :param conv:
        :param N:
        """
        self.X0 = np.random.multivariate_normal(means[0], conv, N)
        self.X1 = np.random.multivariate_normal(means[1], conv, N)
        self.X2 = np.random.multivariate_normal(means[2], conv, N)
        self.N = N
        self.X = np.concatenate((self.X0, self.X1, self.X2), axis= 0)
        self.K = 3
    def plot_data(self):
        plt.plot(self.X0[:, 0], self.X0[:, 1], 'b^', markersize = 4, alpha = .8)
        plt.plot(self.X1[:, 0], self.X1[:, 1], 'go', markersize = 4, alpha = .8)
        plt.plot(self.X2[:, 0], self.X2[:, 1], 'rs', markersize = 4, alpha = .8)
        plt.axis('equal')
        plt.plot()
        plt.show()

if __name__ == '__main__':
    means = [[2, 2], [8, 3], [3, 6]]
    cov = [[1, 0], [0, 1]]
    N = 500
    data = Data(means, cov, N)
    #data.plot_data()
    k_mean = K_mean(data)
    k_mean.training()

