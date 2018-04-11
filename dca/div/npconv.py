import numpy as np
import tensorflow as tf

sess = tf.Session()
inp = np.random.uniform(size=(5, 5))
inpd = np.expand_dims(np.expand_dims(inp, axis=0), axis=-1)
tinp = tf.constant(inpd, dtype=tf.float32)


def singlefilter_valid():
    """Single filter, depth 1, valid padding. WORKS"""
    tconv = tf.layers.Conv2D(filters=1, kernel_size=2, padding="VALID")
    tout = tconv.apply(tinp)
    sess.run(tf.global_variables_initializer())

    toutv, kernel = sess.run((tout, tconv.kernel))
    print(toutv.shape, kernel.shape)
    kernel = kernel.squeeze()
    toutv = toutv.squeeze()
    print("TF after squeeze:", toutv.shape, kernel.shape)

    m, n = kernel.shape
    y, x = inp.shape
    y = y - m + 1
    x = x - m + 1
    out = np.zeros((y, x))
    for i in range(y):
        for j in range(x):
            out[i][j] = np.sum(inp[i:i + m, j:j + m] * kernel)
    # for r in range(3):
    #     data[r,:] = np.convolve(inp[r,:], H_r, 'same')

    # for c in range(3):
    #     data[:,c] = np.convolve(inp[:,c], H_c, 'same')
    print(toutv, "\n", out)
    print(toutv.shape, out.shape)
    print((toutv == out).all())
