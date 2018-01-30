import numpy as np
import tensorflow as tf

sess = tf.Session()

grids = np.random.uniform(size=(2, 3, 3, 4)).astype(np.float32)
oh_cells = np.zeros((2, 3, 3, 1), dtype=np.float32)
oh_cells[0][0][2][0] = 1
oh_cells[1][1][1][0] = 1

inp_grids = tf.constant(grids)
inp_cells = tf.constant(oh_cells)

conv1 = tf.layers.conv2d(
    inputs=inp_grids, filters=4, kernel_size=4, padding="same")
stacked = tf.concat([conv1, inp_cells], axis=3)

sess.run(tf.global_variables_initializer())
npconv = sess.run(conv1)
npstacked = sess.run(stacked)
print(npconv[0])
print(npstacked[0])
