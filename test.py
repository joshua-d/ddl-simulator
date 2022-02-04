import tensorflow as tf
import numpy as np


t1 = tf.constant([1, 2, 3])
t2 = tf.constant([2, 3, 4])

m = tf.reduce_mean([t1, t2])
print(m)