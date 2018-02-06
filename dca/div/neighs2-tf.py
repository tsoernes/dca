import numpy as np
import tensorflow as tf

from grid import RhombusAxialGrid


def prassert(a, b):
    assert type(a) is np.ndarray, type(a)
    assert type(b) is np.ndarray, type(b)
    assert (a.shape == b.shape), (a.shape, b.shape)
    assert (a == b).all(), (a, b)


sess = tf.Session()

cell0 = (3, 4)
cell1 = (2, 1)
cells = [cell0, cell1]
cellneighs0 = RhombusAxialGrid.neighbors(2, *cell0, separate=True, include_self=True)
cellneighs1 = RhombusAxialGrid.neighbors(2, *cell1, separate=True, include_self=True)
ch1 = 13
ch2 = 15
tf_cell = tf.placeholder(shape=[2], dtype=tf.int32)
tf_cells = tf.placeholder(shape=[None, 2], dtype=tf.int32)

np_grid0 = np.random.choice([0, 1], size=(7, 7, 70)).astype(np.bool)
np_grid0[(*cellneighs0, ch1)] = 0
np_grid0[(*cellneighs0, ch2)] = 1
np_grid1 = np.zeros(shape=(7, 7, 70), dtype=np.bool)
np_grids = np.array([np_grid0, np_grid1])
tf_grids = tf.placeholder(shape=[None, 7, 7, 70], dtype=tf.bool)

np_neighs_mask = RhombusAxialGrid.neighbors_all_oh()
neighs_mask = tf.constant(np_neighs_mask, dtype=tf.bool)


def get_elig_np(grid, cell):
    neighs = grid[np.where(np_neighs_mask[cell])]
    np_alloc_map = np.bitwise_or.reduce(neighs)
    np_eligible_chs = np.nonzero(np.invert(np_alloc_map))[0]
    return np_alloc_map, np_eligible_chs


def get_elig_tf(inp):
    # inp.shape: (7, 7, 71)
    grid = inp[:, :, :-1]
    neighs_mask_local = inp[:, :, -1]
    # Code below here needs to be mapped because 'where' will produce
    # variable length result
    neighs_i = tf.where(neighs_mask_local)
    neighs = tf.gather_nd(grid, neighs_i)
    alloc_map = tf.reduce_any(neighs, axis=0)
    # eligible_chs = tf.reshape(tf.where(tf.logical_not(alloc_map)), [-1])
    return alloc_map
    # return eligible_chs


def make_inp():
    neighs_mask_local = tf.gather_nd(neighs_mask, tf_cells)
    inp = tf.concat([tf_grids, tf.expand_dims(neighs_mask_local, axis=3)], axis=3)
    return inp


sess.run(tf.global_variables_initializer())
alloc_maps = sess.run(
    tf.map_fn(get_elig_tf, make_inp(), dtype=tf.bool), {
        tf_cells: cells,
        tf_grids: np_grids
    })
np_alloc_map0, np_eligible_chs0 = get_elig_np(np_grid0, cell0)
np_alloc_map1, np_eligible_chs1 = get_elig_np(np_grid1, cell1)

# print(eligible_chs)
# print(np_grid[:, :, ch1])

prassert(alloc_maps[0], np_alloc_map0)
prassert(alloc_maps[1], np_alloc_map1)
# prassert(eligible_chs, np_eligible_chs)
# assert ch1 in eligible_chs
# assert ch1 in np_eligible_chs
# assert ch2 not in eligible_chs
# assert ch2 not in np_eligible_chs

np_grid1[cell1][3] = 1
np_grids = np.array([np_grid0, np_grid1])
np_qvals = np.random.uniform(size=(2, 70))
qvals = tf.constant(np_qvals)
arange = tf.expand_dims(tf.range(tf.shape(cells)[0]), axis=1)
rcells = tf.concat([arange, cells], axis=1)
alloc_maps = tf.gather_nd(tf_grids[:, :, :, :70], rcells)
inuse_qvals = tf.boolean_mask(qvals, alloc_maps)
res = sess.run(alloc_maps, {tf_cells: cells, tf_grids: np_grids})
print(res.shape)
print(res)
