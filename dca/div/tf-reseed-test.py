# How to run: 'p3 -m div.tf-reseed-test' from dca directory
import tensorflow as tf
from nets.qnet import QNet
import numpy as np


class RandNum:
    def __init__(self):
        # tf.set_random_seed(0)  # Doesn't work
        tf.reset_default_graph()
        # tf.set_random_seed(0)  # Works
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf.set_random_seed(0)  # Works
        self.build()
        # tf.set_random_seed(0)  # Doesn't work
        init = tf.global_variables_initializer()
        # tf.set_random_seed(0)  # Doesn't work
        self.sess.run(init)
        # tf.set_random_seed(0)  # Doesn't work

    def build(self):
        self.a = tf.random_uniform([1])

    def run(self):
        print(self.sess.run(self.a))


tf.set_random_seed(0)  # Doesn't work
n = RandNum()
n.run()
n.sess.close()
m = RandNum()
m.run()

pp = {
    'net_lr': 1e-6,
    'gamma': 0.9,
    'n_channels': 70,
    'batch_size': 1,
    'tfprofiling': False,
    'no_gpu': False,
    'rows': 7,
    'cols': 7,
}
grid = np.ones((7, 7, 70))
cell = (2, 2)
qnet1 = QNet(True, pp, None)
res1 = qnet1.forward(grid, cell)
qnet2 = QNet(True, pp, None)
res2 = qnet2.forward(grid, cell)
print(res1)
print((res1 == res2).all())
