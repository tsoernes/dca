import sys
from time import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.client import timeline

import dataloader
from nets.utils import (get_act_fn_by_name, get_init_by_name,
                        get_optimizer_by_name)


"""
possible data prep: set unused channels to -1,
OR make it unit gaussian. refer to alphago paper -- did they prep
the board? did they use the complete board as input to any of their
nets, or just features?

sanity checks:
- double check that loss is sane
for softmax classifier: print loss, should be roughly:
"-log(1/n_classes)"
- make sure that it's possible to overfit
(ie loss of nearly 0, when no regularization)
a very small portion (eg <20 samples) of the training data

Batch size 8 took 307.63 seconds
Batch size 16 took 163.69 seconds
Batch size 32 took 92.84 seconds
Batch size 64 took 53.94 seconds
Batch size 128 took 38.26 seconds
Batch size 256 took 34.56 seconds
Batch size 512 took 29.68 seconds
Batch size 1024 took 27.80 seconds
Batch size 2048 took 24.63 seconds

Perhaps reducing call rates will increase difference between
fixed/random and a good alg, thus making testing nets easier.
If so then need to retest sarsa-strats and redo hyperparam opt.

"""


class Net:
    def __init__(self, pp, logger, name):
        self.logger = logger
        self.save = pp['save_net']
        restore = pp['restore_net']
        self.gamma = pp['gamma']
        self.batch_size = pp['batch_size']
        self.n_channels = pp['n_channels']
        self.pp = pp
        main_path = "model/" + name
        self.model_path = main_path + "/model.cpkt"
        self.log_path = main_path + "/logs"
        self.quit_sim = False

        self.act_fn = get_act_fn_by_name(pp['act_fn'])
        self.kern_init_conv = get_init_by_name(pp['weight_init_conv'])
        self.kern_init_dense = get_init_by_name(pp['weight_init_dense'])
        self.regularizer = None
        if pp['layer_norm']:
            self.regularizer = tf.contrib.layers.layer_norm

        self.trainer = get_optimizer_by_name(pp['optimizer'], pp['net_lr'])

        tf.reset_default_graph()
        self.options = None
        self.run_metadata = None
        if pp['tfprofiling']:
            self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()
        # tf.logging.set_verbosity(tf.logging.WARN)

        # Allocate only minimum amount necessary GPU memory at start, then grow
        if pp['no_gpu']:
            config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.set_random_seed(pp['rng_seed'])
        self.sess = tf.Session(config=config)

        self.build()
        init = tf.global_variables_initializer()
        if self.save or restore:
            self.saver = tf.train.Saver()
        self.sess.run(init)
        if restore:
            # Could do a try/except and build if loading fails
            self.logger.error(f"Restoring model from {self.model_path}")
            self.saver.restore(self.sess, self.model_path)
        self.data_is_loaded = False

    def _build_base_net(self, grid, cell, name):
        with tf.variable_scope(name):
            conv1 = tf.layers.conv2d(
                inputs=grid,
                filters=self.n_channels,
                kernel_size=4,
                padding="same",
                kernel_initializer=self.kern_init_conv(),
                kernel_regularizer=self.regularizer,
                use_bias=True,  # Default setting
                activation=self.act_fn)
            conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=70,
                kernel_size=3,
                padding="same",
                kernel_initializer=self.kern_init_conv(),
                kernel_regularizer=self.regularizer,
                use_bias=True,
                activation=self.act_fn)
            stacked = tf.concat([conv2, cell], axis=3)
            flat = tf.layers.flatten(stacked)
            return flat

    def _build_default_trainer(self, var_list=None):
        """If var_list is not specified, defaults to GraphKey.TRAINABLE_VARIABLES"""
        if self.pp['max_grad_norm'] is not None:
            gradients, trainable_vars = zip(*self.trainer.compute_gradients(
                self.loss, var_list=var_list))
            clipped_grads, grad_norms = tf.clip_by_global_norm(
                gradients, self.pp['max_grad_norm'])
            do_train = self.trainer.apply_gradients(
                zip(clipped_grads, trainable_vars))
        else:
            do_train = self.trainer.minimize(self.loss, var_list=var_list)
        return do_train

    def load_data(self):
        if self.data_is_loaded:
            return
        data = dataloader.get_data_h5py(self.batch_size)
        # data = self.get_data()
        self.n_train_steps = data['n_train_steps']
        self.n_test_steps = data['n_test_steps']
        self.train_gen = data['train_gen']
        self.test_gen = data['test_gen']
        self.data_is_loaded = True
        self.qvals = np.load("qtable.npy")

    def save_model(self):
        self.logger.error(f"Saving model to path {self.model_path}")
        self.saver.save(self.sess, self.model_path)

    def save_timeline(self):
        if self.pp['tfprofiling']:
            fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open(self.pp['tfprofiling'], 'w') as f:
                f.write(chrome_trace)

    def build(self):
        raise NotImplementedError

    def train(self):
        self.load_data()
        losses = []
        self.logger.warn(f"Training {self.n_train_steps} minibatches of size"
                         f" {self.batch_size} for a total of"
                         f" {self.n_train_steps * self.batch_size} examples")
        for i in range(self.n_train_steps):
            # Get expected returns following a greedy policy from the
            # next state: max a': Q(s', a', w_old)
            data = next(self.train_gen)
            cells = data['cells']
            targets = np.zeros(len(cells), dtype=np.float32)
            actions = data['actions']
            for j, ch in enumerate(actions):
                targets[j] = self.qvals[cells[j]][ch]
            curr_data = {
                self.grid: data['grids'],
                self.cell: data['oh_cells'],
                self.action: actions,
                self.next_q: targets,
                # self.action: data['actions'],
                self.reward: data['rewards'],
                # self.next_grid: data['next_grids'],
                # self.next_cell: data['next_cells']
            }
            _, loss, summary = self.sess.run(
                [self.do_train, self.loss, self.summaries], curr_data)
            if i % 50 == 0:
                # tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # run_metadata = tf.RunMetadata()
                # self.train_writer.add_run_metadata(
                #     run_metadata, 'step%d' % i)
                self.train_writer.add_summary(summary, i)
                self.logger.info(f"Iter {i}\tloss: {loss:.2f}")
                losses.append(loss)
            if False:
                self.logger.debug(f"grid: {data['grids'][0]} \n"
                                  f"cell: {data['cells'][0]} \n"
                                  f"oh_cell: {data['oh_cells'][0]} \n"
                                  f"action: {data['actions'][0]} \n"
                                  f"reward: {data['rewards'][0]} \n"
                                  f"target: {targets[0]} \n"
                                  f"loss: {loss} \n")
                sys.exit(0)
            if np.isnan(loss) or np.isinf(loss):
                self.logger.error(f"Invalid loss: {loss}")
                sys.exit(0)
                break
            if self.quit_sim:
                sys.exit(0)
        if self.save:
            self.save_model()
        # self.eval()
        plt.plot(losses)
        plt.ylabel("Loss")
        plt.xlabel(f"Iterations, in {self.batch_size}s")
        plt.show()

    def eval(self):
        self.load_data()
        self.logger.warn(f"Evaluating {self.n_test_steps} minibatches of size"
                         f" {self.batch_size} for a total of"
                         f"  {self.n_test_steps * self.batch_size} examples")
        eval_losses = []
        for i in range(self.n_test_steps):
            # Get expected returns following a greedy policy from the
            # next state: max a': Q(s', a', w_old)
            data = next(self.test_gen)
            next_data = {
                self.grid: data['next_grids'],
                self.cell: data['next_cells']
            }
            next_q_maxs = self.sess.run(self.q_max, next_data)
            r = data['rewards']
            q_targets = r + self.gamma * next_q_maxs
            curr_data = {
                self.grid: data['grids'],
                self.cell: data['cells'],
                self.action: data['actions'],
                self.target_q: q_targets
            }
            loss, summary = self.sess.run([self.loss, self.summaries],
                                          curr_data)
            self.eval_writer.add_summary(summary, i)
            eval_losses.append(loss)
            if np.isnan(loss) or np.isinf(loss):
                self.logger.error(f"Invalid loss: {loss}")
                break
        self.logger.error(
            f"\nEval results: {sum(eval_losses) / len(eval_losses)}")

    def forward(self, grid, cell):
        raise NotImplementedError

    def bench_batch_size(self):
        for bs in [256, 512, 1024, 2048]:
            self.pp['batch_size'] = bs
            t = time.time()
            self.train()
            self.logger.error(
                f"Batch size {bs} took {time.time()-t:.2f} seconds")

    def rand_uniform(self):
        """Used for checking if random seeds are set/working"""
        r = tf.random_uniform([1])
        ra = self.sess.run(r)
        return ra

    @staticmethod
    def _get_trainable_vars(scope):
        trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        trainable_vars_by_name = {
            var.name[len(scope.name):]: var
            for var in trainable_vars
        }
        return trainable_vars_by_name


if __name__ == "__main__":
    import logging
    logger = logging.getLogger('')
    n = Net(logger)
    n.train()
    # n.eval()
