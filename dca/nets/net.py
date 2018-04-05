import os
import signal
import sys
from time import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt  # noqa
from tensorflow.python.client import timeline

import datahandler
from nets.utils import (build_default_minimizer, get_act_fn_by_name,
                        get_init_by_name)
from nets.convlayers import InPlaneSplit

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # warn
"""
Perhaps reducing call rates will increase difference between
fixed/random and a good alg, thus making testing nets easier.
If so then need to retest sarsa-strats and redo hyperparam opt.
"""


class Net:
    def __init__(self, pp, logger, name):
        self.logger = logger
        self.save = pp['save_net']
        restore = pp['restore_net']
        self.batch_size = pp['batch_size']
        self.grid_split = pp['grid_split']
        self.rows, self.cols, self.n_channels = pp['dims']
        self.pp = pp
        main_path = "model/" + name
        self.model_path = main_path + "/model.cpkt"
        self.log_path = main_path + "/logs"
        self.quit_sim = False
        self.weight_vars = []  # Store all weight variables
        self.weight_names = []  # Store all weight names

        # self.neighs_mask = tf.constant(Grid.neighbors_all_oh(), dtype=tf.bool)
        self.act_fn = get_act_fn_by_name(pp['act_fn'])
        self.kern_init_conv = get_init_by_name(pp['weight_init_conv'], pp)
        self.kern_init_dense = get_init_by_name(pp['weight_init_dense'], pp)
        self.conv_regularizer, self.dense_regularizer = None, None
        if pp['layer_norm']:
            self.conv_regularizer = tf.contrib.layers.layer_norm
            self.dense_regularizer = tf.contrib.layers.layer_norm
        if pp['l2_conv']:
            self.conv_regularizer = tf.contrib.layers.l2_regularizer(pp['l2_scale'])
        if pp['l2_dense']:
            self.dense_regularizer = tf.contrib.layers.l2_regularizer(pp['l2_scale'])

        tf.reset_default_graph()
        self.options = None
        self.run_metadata = None
        if pp['tfprofiling']:
            self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()
        # tf.logging.set_verbosity(tf.logging.WARN)

        if not pp['gpu']:
            # self.logger.error("Not using GPU")
            config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            config = tf.ConfigProto()
        # config.intra_opt_parallelism_threads = 4
        # Allocate only minimum amount necessary GPU memory at start, then grow
        config.gpu_options.allow_growth = True
        tf.set_random_seed(pp['rng_seed'])
        self.sess = tf.Session(config=config)

        loss, trainable_vars = self.build()
        glob_vars = set(tf.global_variables())
        if self.save or restore:
            self.saver = tf.train.Saver(
                # var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'))
                var_list=trainable_vars)
        self.sess.run(tf.global_variables_initializer())
        if restore:
            # Could do a try/except and build if loading fails
            # self.logger.error(f"Restoring model from {self.model_path}")
            self.logger.error(f"Restoring parameters {trainable_vars}")
            self.saver.restore(self.sess, self.model_path)
        if trainable_vars is not None:
            self.do_train, self.lr = build_default_minimizer(
                **pp, loss=loss, var_list=trainable_vars)
        # Initialize trainer variables
        self.sess.run(tf.variables_initializer(set(tf.global_variables()) - glob_vars))

        if pp['train_net']:
            self.data_is_loaded = False
            signal.signal(signal.SIGINT, self.exit_handler)

    def exit_handler(self, *args):
        self.quit_sim = True

    def build(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def _build_base_net2(self, top_inp, cell, name):
        """A series of convolutional layers with 'grid' and 'frep' as inputs,
        and 'cell' stacked with the outputs"""
        # TODO Try one conv after cell stack
        with tf.variable_scope('model/' + name):
            inp = tf.concat([top_inp, cell], axis=3) if self.pp['top_stack'] else top_inp
            nconvs = len(self.pp['conv_nfilters'])
            for i in range(nconvs - 1):
                conv = InPlaneSplit(
                    self.pp['conv_kernel_sizes'][i],
                    stride=1,
                    use_bias=self.pp['conv_bias'],
                    padding="SAME",
                    kernel_initializer=self.conv_regularizer)
                islast = i == nconvs - 2
                inp = conv.apply(inp, islast)
            inp = self.add_conv_layer(inp, self.pp['conv_nfilters'][nconvs - 1],
                                      self.pp['conv_kernel_sizes'][nconvs - 1])
            self.logger.error(f"Conv out shape, before cellstack: {inp.shape}")
            conv_out = inp if self.pp['top_stack'] else tf.concat([inp, cell], axis=3)
            out = tf.layers.flatten(conv_out)
            return out

    def _build_base_net(self, top_inp, cell, name):
        """A series of convolutional layers with 'grid' and 'frep' as inputs,
        and 'cell' stacked with the outputs"""
        # TODO Try one conv after cell stack
        with tf.variable_scope('model/' + name):
            inp = tf.concat([top_inp, cell], axis=3) if self.pp['top_stack'] else top_inp
            for i in range(len(self.pp['conv_nfilters'])):
                inp = self.add_conv_layer(inp, self.pp['conv_nfilters'][i],
                                          self.pp['conv_kernel_sizes'][i])
            conv_out = inp if self.pp['top_stack'] else tf.concat([inp, cell], axis=3)
            out = tf.layers.flatten(conv_out)
            return out

    def add_conv_layer(self, inp, nfilters, kernel_size, padding="same", use_bias=None):
        use_bias = self.pp['conv_bias'] if use_bias is None else use_bias
        conv = tf.layers.Conv2D(
            filters=nfilters,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=self.kern_init_conv(),
            kernel_regularizer=self.conv_regularizer,
            use_bias=use_bias,
            activation=self.act_fn)
        out = conv.apply(inp)
        self.weight_vars.append(conv.kernel)
        self.weight_names.append(conv.kernel.name)
        if self.pp['conv_bias']:
            self.weight_vars.append(conv.bias)
            self.weight_names.append(conv.bias.name)
        return out

    def add_dense_layer(self, inp, nunits, kern_init, act_fn=None):
        dense = tf.layers.Dense(
            units=nunits,
            kernel_initializer=kern_init,
            kernel_regularizer=self.dense_regularizer,
            use_bias=False,
            activation=act_fn)
        out = dense.apply(tf.layers.flatten(inp))
        self.weight_vars.append(dense.kernel)
        return out

    def load_data(self):
        if self.data_is_loaded:
            return
        data = datahandler.get_data_h5py(self.batch_size)
        self.n_train_steps = data['n_train_steps']
        self.n_test_steps = data['n_test_steps']
        self.train_gen = data['train_gen']
        self.test_gen = data['test_gen']
        self.qvals = np.load("qtable.npy")
        self.data_is_loaded = True

    def save_model(self):
        inp = ""
        path = n_path = self.model_path
        if os.path.isdir(path + ".index"):
            while inp not in ["Y", "N", "A"]:
                inp = input("A model exists in {path}. Overwrite (Y), Don't save (N), "
                            "or Save to directory (A): ").upper()
            if inp == "A":
                i = 0
                while os.path.isdir(n_path):
                    n_path = path + str(i)
                    i += 1
        if inp not in ["N"]:
            if not os.path.exists("model"):
                os.mkdir("model")
            self.logger.error(f"Saving model to path {n_path}")
            self.saver.save(self.sess, n_path)

    def save_timeline(self):
        if self.pp['tfprofiling']:
            fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open(self.pp['tfprofiling'], 'w') as f:
                f.write(chrome_trace)

    def train(self):
        self.load_data()
        losses = []
        n_train_steps = min(
            self.n_train_steps,
            self.pp['train_net']) if self.pp['train_net'] > 1 else self.n_train_steps
        self.logger.warn(f"Training {n_train_steps} minibatches of size"
                         f" {self.batch_size} for a total of"
                         f" {self.n_train_steps * self.batch_size} examples")
        for i in range(n_train_steps):
            data = next(self.train_gen)
            grids = data['grids']
            cells = data['cells']
            # q_targets = np.zeros(len(cells), dtype=np.float32)
            chs = data['chs']
            # for j, ch in enumerate(chs):
            #     q_targets[j] = self.qvals[cells[j]][ch]
            if False:
                self.logger.error(f"grid: {data['grids'][0]} \n"
                                  f"cell: {data['cells'][0]} \n"
                                  f"action: {data['chs'][0]} \n"
                                  f"reward: {data['rewards'][0]} \n")
                # f"target: {q_targets[0]} \n")
                sys.exit(0)
            # NOTE TODO NOTE
            # qnet backward has changed. it now takes target_q instead of next_g
            # so target_q = reward + gamma * next_q has to be calculated here.
            # loss, lr, td_err = self.backward_supervised(
            #     grids=grids, cells=cells, chs=chs, q_targets=q_targets)
            loss, lr, td_err = self.backward(
                grids=grids,
                cells=cells,
                chs=chs,
                rewards=data['rewards'],
                next_grids=data['next_grids'],
                next_cells=data['next_cells'],
                gamma=self.pp['gamma'])
            self.sess.run(self.copy_online_to_target)
            if (i * self.batch_size) % 100 == 0:
                # tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # run_metadata = tf.RunMetadata()
                # self.train_writer.add_run_metadata(
                #     run_metadata, 'step%d' % i)
                self.logger.info(f"Iter {i}\tloss: {loss:.2f}")
                losses.append(loss)
            if np.isnan(loss) or np.isinf(loss):
                self.logger.error(f"Invalid loss: {loss}")
                sys.exit(0)
                break
            if self.quit_sim:
                if self.save:
                    inp = ""
                    while inp not in ["Y", "N"]:
                        inp = input("Premature exit. Save? Y/N").upper()
                    if inp == "Y":
                        self.save_model()
                    sys.exit(0)
                else:
                    sys.exit(0)
        if self.save:
            self.save_model()
        # plt.plot(losses)
        # plt.ylabel("Loss")
        # plt.xlabel(f"Iterations, in {self.batch_size}s")
        # plt.show()

    def bench_batch_size(self):
        for bs in [256, 512, 1024, 2048]:
            self.pp['batch_size'] = bs
            t = time.time()
            self.train()
            self.logger.error(f"Batch size {bs} took {time.time()-t:.2f} seconds")

    def rand_uniform(self):
        """Used for checking if random seeds are set/working"""
        r = tf.random_uniform([1])
        ra = self.sess.run(r)
        return ra

    def get_weights(self):
        w = self.sess.run(self.weight_vars)
        return w, self.weight_names

    def inuse_qvals(self, grids, cells, qvals):
        """
        Tested and works, but slower than CPU

        Return a dense array of q-values that are in use

        Expects:
        grids.shape: [None, 7, 7, 70/140]
        cells.shape: [None, 2]
        qvals.shape: [None, 70]
        """
        arange = tf.expand_dims(tf.range(tf.shape(cells)[0]), axis=1)
        rcells = tf.concat([arange, cells], axis=1)
        alloc_maps = tf.gather_nd(grids[:, :, :, :70], rcells)
        inuse_qvals = tf.cast(alloc_maps, tf.float32) * qvals
        return inuse_qvals

    def eligible_qvals(self, grids, cells, qvals):
        """
        Tested and works, but slower than CPU

        Return a dense array of q-values that are eligible to assignment
        without violating the reuse constraint

        Expects:
        grids.shape: [None, 7, 7, 70/140]
        cells.shape: [None, 2]
        qvals.shape: [None, 70]
        """

        def get_elig_alloc_map(inp):
            # inp.shape: (7, 7, 71)
            grid = inp[:, :, :70]
            neighs_mask_local = inp[:, :, -1]
            # Code below here needs to be mapped because 'where' will produce
            # variable length result
            neighs_i = tf.where(neighs_mask_local)
            neighs = tf.gather_nd(grid, neighs_i)
            alloc_map = tf.reduce_any(neighs, axis=0)
            # Can't return elig chs because variable length result
            # eligible_chs = tf.reshape(tf.where(tf.logical_not(alloc_map)), [-1])
            return alloc_map

        neighs_mask_local = tf.gather_nd(self.neighs_mask, cells)
        inp = tf.concat([grids, tf.expand_dims(neighs_mask_local, axis=3)], axis=3)
        alloc_maps = tf.map_fn(get_elig_alloc_map, inp, dtype=tf.bool)
        # TODO This should be sparse instead, allowing for multiple grids
        # in forward batch or for this method to be used in a backward pass
        # elig_qvals = tf.boolean_mask(qvals, tf.logical_not(alloc_maps))
        elig_qvals = tf.cast(tf.logical_not(alloc_maps), tf.float32) * qvals
        return elig_qvals


if __name__ == "__main__":
    import logging
    logger = logging.getLogger('')
    n = Net(logger)
    n.train()
