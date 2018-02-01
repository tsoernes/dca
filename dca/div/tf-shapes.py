import numpy as np
import tensorflow as tf

from grid import RhombusAxialGrid

sess = tf.Session()

grids = np.random.uniform(size=(2, 3, 3, 4)).astype(np.float32)
oh_cells = np.zeros((2, 3, 3, 1), dtype=np.float32)
oh_cells[0][0][2][0] = 1
oh_cells[1][1][1][0] = 1

inp_grids = tf.constant(grids)
inp_cells = tf.constant(oh_cells)

conv1 = tf.layers.conv2d(inputs=inp_grids, filters=4, kernel_size=4, padding="same")
stacked = tf.concat([conv1, inp_cells], axis=3)

sess.run(tf.global_variables_initializer())
npconv = sess.run(conv1)
npstacked = sess.run(stacked)
# Works as expected:
# print(npconv[0])
# print(npstacked[0])

########################
# On gathering neighbors ..

grid = np.random.uniform(size=(7, 7, 3)).astype(np.float32)
neighs2i_sep = RhombusAxialGrid.neighbors(2, 2, 3, separate=True)
neighs2i = RhombusAxialGrid.neighbors(2, 2, 3, separate=False)
tf_grid = tf.constant(grid)
tf_neighs2i = tf.constant(neighs2i)
tf_neighs = sess.run(tf.gather_nd(tf_grid, tf_neighs2i))
# OK
print((tf_neighs == grid[neighs2i_sep]).all())

# With batches
grids = np.random.uniform(size=(2, 7, 7, 3)).astype(np.float32)
tf_grids = tf.constant(grids)
neighs = grids[(np.repeat(0, len(neighs2i_sep[0])), *neighs2i_sep)]
tf_neighs2 = sess.run(tf.gather_nd(tf_grids[0], tf_neighs2i))
# OK
print((tf_neighs2 == neighs).all())
