import tensorflow as tf
import numpy as np

sess = tf.Session()

a = tf.zeros(0)
b = tf.placeholder(dtype=tf.int64)
fn = lambda: tf.argmax(b)
fn2 = lambda: fn()

res = tf.cond(
    tf.equal(tf.size(a), tf.constant(0)), lambda: tf.constant(-1, dtype=tf.int64), fn2)

res2 = tf.cond(
    tf.equal(tf.size(a), tf.constant(0)), lambda: tf.constant(-1, dtype=tf.int64),
    lambda: tf.add(fn(), tf.constant(1, dtype=tf.int64)))

print(sess.run(res, {b: np.zeros(0, dtype=np.int64)}))
