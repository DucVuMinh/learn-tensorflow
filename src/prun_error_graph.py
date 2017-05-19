import tensorflow as tf

g = tf.Graph()
# add ops to the default graph
a = tf.constant(3)
# add ops to the user created graph
with g.as_default():
    b = tf.constant(5)
