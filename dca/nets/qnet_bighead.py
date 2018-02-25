import numpy as np
import tensorflow as tf
from tensorflow import bool as boolean
from tensorflow import float32, int32

from nets.utils import get_act_fn_by_name, get_init_by_name, prep_data_grids

np.set_printoptions(threshold=np.nan)


class BigHeadQNet():
    def __init__(self, name, pp, logger):
        self.logger = logger
        self.grid_split = pp['grid_split']
        self.rows, self.cols, self.n_channels = pp['dims']
        self.act_fn = get_act_fn_by_name(pp['act_fn'])
        self.kern_init_conv = get_init_by_name(pp['weight_init_conv'])
        self.kern_init_dense = get_init_by_name(pp['weight_init_dense'])
        self.pp = pp
        self.quit_sim = False
        self.i = 0

        tf.reset_default_graph()
        config = tf.ConfigProto()
        tf.set_random_seed(pp['rng_seed'])
        self.sess = tf.Session(config=config)

        self.lr = tf.constant(pp['net_lr'])
        self.build()
        self.sess.run(tf.global_variables_initializer())
        self.copy_online_to_target = tf.no_op()

    def rand_uniform(self):
        """Used for checking if random seeds are set/working"""
        r = tf.random_uniform([1])
        ra = self.sess.run(r)
        return ra

    def save_model(self):
        pass

    def save_timeline(self):
        pass

    def build(self):
        gridshape = [None, self.rows, self.cols, self.n_channels * 2]
        self.grids = tf.placeholder(boolean, gridshape, "grid")
        self.cells = tf.placeholder(int32, [None, 2], "cell")
        self.chs = tf.placeholder(int32, [None], "ch")
        self.q_targets = tf.placeholder(float32, [None], "qtarget")

        self.grids_f = tf.cast(self.grids, float32)
        self.nrange = tf.range(tf.shape(self.grids)[0])
        self.ncells = tf.concat([tf.expand_dims(self.nrange, axis=1), self.cells], axis=1)
        self.numbered_chs = tf.stack([self.nrange, self.chs], axis=1)
        self.conv1 = tf.layers.conv2d(
            inputs=self.grids_f,
            filters=140,
            kernel_size=5,
            padding="same",
            kernel_initializer=self.kern_init_conv(),
            use_bias=False,  # Default setting
            activation=self.act_fn)
        self.conv2 = tf.layers.conv2d(
            inputs=self.conv1,
            filters=140,
            kernel_size=3,
            padding="same",
            kernel_initializer=self.kern_init_conv(),
            use_bias=False,
            activation=self.act_fn)
        self.conv3 = tf.layers.conv2d(
            inputs=self.conv2,
            filters=70,
            kernel_size=1,
            padding="same",
            kernel_initializer=self.kern_init_conv(),
            use_bias=False,
            activation=self.act_fn)
        self.online_q_vals = tf.gather_nd(self.conv3, self.ncells)

        # Maximum valued ch from online network
        self.online_q_amax = tf.argmax(
            self.online_q_vals, axis=1, name="online_q_amax", output_type=int32)
        # Target Q-value for greedy channel as selected by online network
        self.numbered_q_amax = tf.stack([self.nrange, self.online_q_amax], axis=1)
        self.target_q_max = tf.gather_nd(self.online_q_vals, self.numbered_q_amax)
        # Online Q-value for given ch
        self.online_q_selected = tf.gather_nd(self.online_q_vals, self.numbered_chs)

        self.loss = tf.losses.mean_squared_error(
            labels=tf.stop_gradient(self.q_targets), predictions=self.online_q_selected)
        trainer = tf.train.MomentumOptimizer(self.pp['net_lr'], momentum=0.95)
        self.do_train = trainer.minimize(self.loss)

    def forward(self, grid, cell, ce_type, frep=None):
        data = {
            self.grids: prep_data_grids(grid, split=self.grid_split),
            self.cells: [cell]
        }
        q_vals = self.sess.run(self.online_q_vals, feed_dict=data)[0]
        assert q_vals.shape == (self.n_channels, ), f"{q_vals.shape}\n{q_vals}"
        return q_vals

    def backward(self,
                 grids,
                 cells,
                 chs,
                 rewards,
                 next_grids,
                 next_cells,
                 gamma,
                 freps=None,
                 next_freps=None,
                 next_chs=None,
                 weights=None) -> (float, float):
        bdata = {
            self.grids: prep_data_grids(next_grids, self.grid_split),
            self.cells: [next_cells]
        }
        next_qvals = self.sess.run(self.target_q_max, bdata)
        q_targets = rewards + gamma * next_qvals

        data = {
            self.grids: prep_data_grids(grids, self.grid_split),
            self.chs: chs,
            self.q_targets: q_targets,
            self.cells: [cells]
        }
        # if self.i == 0 or self.i == 1000:
        #     dvars = [
        #         self.grids_f, self.nrange, self.ncells, self.numbered_chs, self.conv1,
        #         self.conv2, self.conv3, self.online_q_vals, self.online_q_amax,
        #         self.numbered_q_amax, self.target_q_max, self.online_q_selected
        #     ]
        #     out = self.sess.run(dvars, data)
        #     self.logger.error("\n\n\n PRINTING \n\n\n")
        #     self.logger.error(out)
        _, loss, lr = self.sess.run([self.do_train, self.loss, self.lr], feed_dict=data)
        self.i += 1
        return loss, lr, 1
