import sys
from time import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.client import timeline

import dataloader
"""
consider batch norm [ioffe and szegedy, 2015]
batch norm is inserted after fully connected or convolutional
layers and before nonlinearity

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

On finding learning rate:
start very small (eg 1e-6), make sure loss is barely changing
or decreasing very slowly.

on tuning hyperparams:
if cost goes over 3x original cost, break out early

big gap between train and test accuracy:
overfitting. reduce net size or increase regularization
no gap: increase net size

debugging nets: track ratio of weight updates/weight magnitues
should be somewhere around 0.001 or so. if too high, decrease
learning rate, if too log (like 1e-6), increase lr.

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

TODO Reproducible results
tf.set_random_seed(1)  # Do in numpy for call generation also
"""


class Net:
    def __init__(self, pp, logger, name, restore=True, save=True):
        self.logger = logger
        self.save = save
        self.alpha = pp['net_lr']
        self.gamma = pp['gamma']
        self.batch_size = pp['batch_size']
        self.n_channels = pp['n_channels']
        self.pp = pp
        main_path = "model/" + name
        self.model_path = main_path + "/model.cpkt"
        self.log_path = main_path + "/logs"

        if pp['tfprofiling']:
            self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()
        else:
            self.options = None
            self.run_metadata = None
        tf.logging.set_verbosity(tf.logging.WARN)
        tf.reset_default_graph()
        if pp['no_gpu']:
            config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.Session(config=config)
        else:
            self.sess = tf.Session()
        self.build()
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init)
        self.sess.run(self.copy_online_to_target)
        if restore:
            # Could do a try/except and build if loading fails
            self.logger.error(f"Restoring model from {self.model_path}")
            self.saver.restore(self.sess, self.model_path)
        self.data_is_loaded = False

    def load_data(self):
        if self.data_is_loaded:
            return
        data = dataloader.get_data_h5py()
        # data = self.get_data()
        self.n_train_steps = data['n_train_steps']
        self.n_test_steps = data['n_test_steps']
        self.train_gen = data['train_gen']
        self.test_gen = data['test_gen']
        self.data_is_loaded = True

    def save_model(self):
        if self.save:
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

    @staticmethod
    def prep_data_grids(grids, empty_neg=True):
        """
        empty_neg: Represent empty channels as -1 instead of 0
        """
        assert type(grids) == np.ndarray
        if grids.ndim == 3:
            grids = np.expand_dims(grids, axis=0)
        grids.shape = (-1, 7, 7, 70)
        if empty_neg:
            grids = grids.astype(np.int8)
            # Make empty cells -1 instead of 0.
            # Temporarily convert to int8 to save memory
            grids = grids * 2 - 1
        grids = grids.astype(np.float16)
        return grids

    @staticmethod
    def prep_data_cells(cells):
        if type(cells) == tuple:
            cells = [cells]
        oh_cells = np.zeros((len(cells), 7, 7), dtype=np.float16)
        # One-hot grid encoding
        for i, cell in enumerate(cells):
            oh_cells[i][cell] = 1
        oh_cells.shape = (-1, 7, 7, 1)
        # Should not be used when predicting, but could save mem when training
        # del cells

        return oh_cells

    @staticmethod
    def prep_data(grids, cells, actions, rewards, next_grids, next_cells):
        assert type(actions) == np.ndarray
        assert type(rewards) == np.ndarray
        actions = actions.astype(np.int32)
        rewards = rewards.astype(
            np.float32)  # Needs to be 32-bit, else will overflow

        grids = Net.prep_data_grids(grids)
        next_grids = Net.prep_data_grids(next_grids)
        oh_cells = Net.prep_data_cells(cells)
        next_oh_cells = Net.prep_data_cells(next_cells)
        return grids, oh_cells, actions, rewards, next_grids, next_oh_cells

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
            curr_data = {
                self.grid: data['grids'],
                self.cell: data['cells'],
                self.action: data['actions'],
                self.reward: data['rewards'],
                self.next_grid: data['next_grids'],
                self.next_cell: data['next_cells']
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
            if np.isnan(loss) or np.isinf(loss):
                self.logger.error(f"Invalid loss: {loss}")
                sys.exit(0)
                break
        self.save_model()
        self.eval()
        if False:
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


if __name__ == "__main__":
    import logging
    logger = logging.getLogger('')
    n = Net(logger)
    n.train()
    # n.eval()
