import numpy as np
import tensorflow as tf
import pickle
with open("neighdata", "rb") as f:
    data = pickle.load(f)

chgrids, cells, targets = zip(*data)
chgrids, targets = np.array(chgrids), np.array(targets)
oh_cells = np.zeros_like(chgrids)
for i, cell in enumerate(cells):
    oh_cells[i][cell] = 1
chgrids.shape, oh_cells.shape = (-1, 7, 7, 1), (-1, 7, 7, 1)
chgridsneg = chgrids * 2 - 1
targets.shape = (-1, 1)
chgridsneg = chgridsneg.astype(np.float32)
oh_cells = oh_cells.astype(np.float32)
targets = targets.astype(np.float32)
input_grid = chgridsneg[100:102]
input_cell = oh_cells[100:102]
labels = targets[100:102]
print("input_grid:\n", input_grid)
print("\ninput_cell:\n", input_cell)
print("\ninput_labels:\n", labels)

tf.logging.set_verbosity(tf.logging.INFO)
tf.reset_default_graph()
sess = tf.Session()

tfinput_grid = tf.placeholder(
    shape=[None, 7, 7, 1], dtype=tf.float16, name="input_grid")
tfinput_cell = tf.placeholder(
    shape=[None, 7, 7, 1], dtype=tf.float16, name="input_cell")
tflabels = tf.placeholder(shape=[None, 1], dtype=tf.float16)

input_stacked = tf.concat([tfinput_grid, tfinput_cell], axis=3)
print(input_stacked.shape)
conv1even = tf.layers.conv2d(
    inputs=input_stacked,
    filters=1,
    kernel_size=5,
    strides=[1, 2],  # stride every even column
    padding="same",  # pad with 0's
    activation=tf.nn.relu)
conv1odd = tf.layers.conv2d(
    inputs=input_stacked[:, :, 1:],
    filters=1,
    kernel_size=5,
    strides=[1, 2],  # stride every odd column
    padding="same",  # pad with 0's
    activation=tf.nn.relu)
# TODO NEED to revisit the convolution OP math.
# Can I construct an example by hand how this is supposed to work?
# Also check is alternate convs really making a difference or is
# that due to the increased learning rate?

# NOTE PS TODO Data is wrong? because it doesnt check
# whether or not a ch is free in its neighs, it checks
# whether its inuse in cell vs free in neighs
print(conv1even.shape)
print(conv1odd.shape)
# Pad to get same amount of even and odd columns, which allows for
# stacking
conv1odd_pad = tf.pad(conv1odd, [[0, 0], [0, 0], [0, 1], [0, 0]])
# Interleave even and odd columns to one array
# conv1s = tf.stack((conv1even, conv1odd_pad), axis=2)
print(conv1odd_pad.shape)
conv1s = tf.stack((conv1even, conv1odd_pad), axis=3)
print(conv1s.shape)
conv1 = tf.reshape(conv1s, [-1, 7, 8, 1])[:, :, :-1]  # Remove pad col
print(conv1.shape)

conv1_flat = tf.contrib.layers.flatten(conv1)
logits = tf.layers.dense(
    inputs=conv1_flat, units=1)
# Probability of channel not being free for assignment,
# i.e. it is in use in cell or its neighs2
inuse = tf.nn.sigmoid(logits, name="sigmoid_tensor")
loss = tf.losses.sigmoid_cross_entropy(
    tflabels,
    logits=logits)

init = tf.global_variables_initializer()
sess.run(init)
res = sess.run(
    [input_stacked, conv1even, conv1odd, conv1odd_pad, conv1s, conv1,
     conv1_flat, logits, inuse, loss],
    {tfinput_grid: input_grid,
     tfinput_cell: input_cell,
     tflabels: labels})
for val in res:
    print(f"\n{val}\n")
