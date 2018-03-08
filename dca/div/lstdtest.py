import numpy as np
import tensorflow as tf

d = 7 * 7 * 71
reward = 4
gamma = 0.9
frepf = tf.random_uniform(shape=[d, 1], dtype=tf.float32)
next_frepf = tf.random_uniform(shape=[d, 1], dtype=tf.float32)
a_inv = tf.Variable(0.1 * tf.ones(shape=[d, d], dtype=tf.float32))
b = tf.Variable(tf.zeros(shape=[d, 1], dtype=tf.float32))
k = frepf - gamma * next_frepf
v = tf.matmul(tf.transpose(a_inv), k)
c1 = tf.matmul(tf.matmul(tf.transpose(a_inv), frepf), tf.transpose(v))
c2 = 1 + tf.matmul(tf.transpose(v), frepf)
update_a_inv = a_inv.assign_sub(c1 / c2)
update_b = b.assign_add(reward * frepf)
theta = tf.matmul(a_inv, b)
value = tf.matmul(tf.transpose(theta), frepf)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(update_a_inv)
sess.run(update_b)
