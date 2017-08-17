"""
Created by DucVu
Following the tutorial from tensorflow.org
"""

import  tensorflow as tf

q = tf.FIFOQueue(3, dtypes = tf.float16)
init = q.enqueue_many( ([1,2,3],) )
x = q.dequeue()
y = x + 1
q_inc = q.enqueue([y])

init.run()
q_inc.run()
q_inc.run()
q_inc.run()
q_inc.run()
