"""
Fucking guy :)))
"""


import tensorflow as tf
with tf.variable_scope("foo"):
    with tf.name_scope("bar"):
        v = tf.get_variable("v", [1])
        x = 1.0 + v
assert v.name == "foo/v:0"
assert x.op.name == "foo/bar/add"
print x.name
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
