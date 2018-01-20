import tensorflow as tf


class Net:
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


# tf.set_random_seed(0)  # Doesn't work
n = Net()
n.run()
n.run()
m = Net()
m.run()
m.run()
