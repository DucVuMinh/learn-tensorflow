"""
created by DucVM,
Jush show, how a constant be stored in a graph of tensorflow
"""
import tensorflow as tf


graph1 = tf.Graph()
grahp2 = tf.get_default_graph()

with grahp2.as_default():
    my_constant = tf.constant([1,2], dtype= tf.int8)
    my_constant2 = tf.constant([3,4], dtype= tf.int8)
    add = tf.add(my_constant, my_constant2)

with graph1.as_default():
    my_constant3 = tf.constant([5,6], dtype= tf.int8)

with tf.Session(graph =graph1) as sess:
    writer_board = tf.summary.FileWriter("../data/summary/constant_graph", sess.graph)
    print ( sess.run(my_constant3) )
