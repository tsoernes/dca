import numpy as np
import tensorflow as tf
data = np.load("data-freechs.npy")[999:1000]

grids, cells, targets = zip(*data)
grids, targets = np.array(grids), np.array(targets)
oh_cells = np.zeros((len(grids), 7, 7), dtype=np.float16)
for i, cell in enumerate(cells):
    oh_cells[i][cell] = 1
h_targets = np.zeros((len(grids), 70), dtype=np.float16)
for i, targ in enumerate(targets):
    for ch in targ:
        h_targets[i][ch] = 1
grids.shape = (-1, 7, 7, 70)
oh_cells.shape = (-1, 7, 7, 1)
gridsneg = grids * 2 - 1  # Make empty cells -1 instead of 0
chgridsneg = gridsneg.astype(np.float16)
oh_cells = oh_cells.astype(np.float16)
input_grid = chgridsneg[:2]
input_cell = oh_cells[:2]
labels = h_targets[:2]
print("input_grid:\n", input_grid)
print("\ninput_cell:\n", input_cell)
print("\ninput_labels:\n", labels)

tf.logging.set_verbosity(tf.logging.INFO)
tf.reset_default_graph()
sess = tf.Session()

tfinput_grid = tf.placeholder(
    shape=[None, 7, 7, 70], dtype=tf.float16, name="input_grid")
tfinput_cell = tf.placeholder(
    shape=[None, 7, 7, 1], dtype=tf.float16, name="input_cell")
tflabels = tf.placeholder(shape=[None, 70], dtype=tf.float16)

input_stacked = tf.concat([tfinput_grid, tfinput_cell], axis=3)
print(input_stacked.shape)
conv1 = tf.layers.conv2d(
    inputs=input_stacked,
    filters=70,
    kernel_size=5,
    strides=1,
    padding="same",  # pad with 0's
    activation=tf.nn.relu)
conv2 = tf.layers.conv2d(
    inputs=conv1,
    filters=70,
    kernel_size=1,
    strides=1,
    padding="valid",
    activation=tf.nn.relu)
conv2_flat = tf.contrib.layers.flatten(conv2)
logits = tf.layers.dense(
    inputs=conv2_flat, units=70)
prob_inuse = tf.nn.sigmoid(logits, name="sigmoid_tensor")
inuse = tf.greater(prob_inuse, tf.constant(0.5, dtype=tf.float16))

loss = tf.losses.sigmoid_cross_entropy(
    tflabels,
    logits=logits,
    reduction=tf.losses.Reduction.SUM)
print("conv1: ", conv1.shape)
print("conv2: ", conv2.shape)
print("conv2f: ", conv2_flat.shape)
print("logits: ", logits.shape)
print("prob-in: ", prob_inuse.shape)
print("inuse: ", inuse.shape)

init = tf.global_variables_initializer()
sess.run(init)
res = sess.run(
    [input_stacked, conv1, conv2, logits, prob_inuse, loss,
     tf.shape(input_stacked), tf.shape(conv1), tf.shape(conv2),
     tf.shape(logits), tf.shape(prob_inuse)],
    {tfinput_grid: input_grid,
     tfinput_cell: input_cell,
     tflabels: labels})
for val in res:
    print(f"\n{val}\n")
