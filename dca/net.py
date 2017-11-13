import sys

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from matplotlib import pyplot as plt

from utils import BackgroundGenerator
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
"""


class Net:
    def __init__(self, pp, logger, restore=True, save=True):
        self.logger = logger
        self.save = save
        self.alpha = pp['net_lr']
        self.gamma = pp['gamma']
        self.batch_size = pp['batch_size']
        self.pp = pp
        main_path = "model/qnet03"
        self.model_path = main_path + "/model.cpkt"
        self.log_path = main_path + "/logs"

        if pp['tfprofiling']:
            self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()
        else:
            self.options = None
            self.run_metadata = None
        tf.logging.set_verbosity(tf.logging.INFO)
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
        if restore:
            # Could do a try/except and build if loading fails
            self.logger.error(f"Restoring model from {self.model_path}")
            self.saver.restore(self.sess, self.model_path)
        self.data_is_loaded = False

    def load_data(self):
        if self.data_is_loaded:
            return
        data = self.get_data_h5py()
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
        self.input_grid = tf.placeholder(
            shape=[None, 7, 7, 70], dtype=tf.float32, name="input_grid")
        self.input_cell = tf.placeholder(
            shape=[None, 7, 7, 1], dtype=tf.float32, name="input_cell")
        # TODO These are s, not target actions s'.
        self.target_action = tf.placeholder(
            shape=[None], dtype=tf.int32, name="target_action")
        self.target_q = tf.placeholder(
            shape=[None], dtype=tf.float32, name="target_q")

        # conv1 = tf.layers.conv2d(
        #     inputs=self.input_grid,
        #     filters=70,
        #     kernel_size=5,
        #     padding="same",
        #     activation=tf.nn.relu)
        conv1 = tf.contrib.layers.conv2d_in_plane(
            inputs=self.input_grid,
            kernel_size=5,
            stride=1,
            padding="SAME",
            activation_fn=tf.nn.relu)
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=140,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu)
        stacked = tf.concat([conv2, self.input_cell], axis=3)
        conv2_flat = tf.layers.flatten(stacked)

        # Perhaps reducing call rates will increase difference between
        # fixed/random and a good alg, thus making testing nets easier.
        # If so then need to retest sarsa-strats and redo hyperparam opt.
        dense = tf.layers.dense(
            inputs=conv2_flat, units=128, name="dense", activation=tf.nn.relu)
        self.q_vals = tf.layers.dense(
            inputs=conv2_flat, units=70, name="q_vals")
        self.q_amax = tf.argmax(self.q_vals, axis=1, name="q_amax")

        flat_q_vals = tf.reshape(self.q_vals, [-1])
        some_range = tf.range(tf.shape(self.q_vals)[0]) * tf.shape(
            self.q_vals)[1]
        flat_amax = self.q_amax + tf.cast(some_range, tf.int64)
        self.q_max = tf.gather(flat_q_vals, flat_amax)

        flat_target_action = self.target_action + tf.cast(some_range, tf.int32)
        self.q_pred = tf.gather(
            flat_q_vals, flat_target_action, name="action_q_vals")

        # q scores for actions which we know were selected in the given state.
        # q_pred = tf.reduce_sum(q_t * tf.one_hot(actions, num_actions), 1)

        # Below we obtain the loss by taking the sum of squares
        # difference between the target and prediction Q values.
        self.loss = tf.losses.mean_squared_error(
            labels=self.target_q, predictions=self.q_pred)
        # trainer = tf.train.AdamOptimizer(learning_rate=self.alpha)
        # trainer = tf.train.GradientDescentOptimizer(learning_rate=self.alpha)
        trainer = tf.train.RMSPropOptimizer(learning_rate=self.alpha)
        self.do_train = trainer.minimize(self.loss)

        with tf.name_scope("summaries"):
            tf.summary.scalar("learning_rate", self.alpha)
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("qvals", self.q_vals)
        self.summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.log_path + '/train',
                                                  self.sess.graph)
        self.eval_writer = tf.summary.FileWriter(self.log_path + '/eval')

        self.shapes = \
            [tf.shape(stacked),
             tf.shape(conv1),
             tf.shape(conv2_flat),
             tf.shape(dense),
             tf.shape(self.q_vals),
             tf.shape(self.q_amax),
             tf.shape(flat_q_vals),
             tf.shape(flat_amax),
             tf.shape(self.q_max),
             tf.shape(flat_target_action),
             tf.shape(self.q_pred),
             tf.shape(self.loss)]

        # TODO If possible, do epsilon-greedy action selection in TF.
        # Should reduce the amount of data passed between CPU/GPU.
        # TODO Calculate loss in graph
        # q_target = reward + gamma * q_max
        # td_error = q_pred - tf.stop_gradient(q_target)
        # TODO Reproducible results
        # tf.set_random_seed(1)  # Do in numpy for call generation also
        # TODO Keep separate weights for target Q network
        # update_target_fn will be called periodically to copy Q network to target Q network
        # q_func_vars = scope_vars(absolute_scope_name("q_func"))
        # target_q_func_vars = scope_vars(absolute_scope_name("target_q_func"))
        # update_target_expr = []
        # for var, var_target in zip(
        #         sorted(q_func_vars, key=lambda v: v.name),
        #         sorted(target_q_func_vars, key=lambda v: v.name)):
        #     update_target_expr.append(var_target.assign(var))
        # update_target_expr = tf.group(*update_target_expr)

    def get_data_h5py(self):
        # Open file handle, but don't load contents into memory
        h5f = h5py.File("data-experience.0.hdf5", "r")
        entries = len(h5f['grids'])

        split_perc = 0.9  # Percentage of data to train on
        split = int(entries * split_perc) // self.batch_size
        end = entries // self.batch_size

        def data_gen(start, stop):
            for i in range(start, stop):
                batch = slice(i * self.batch_size, (i + 1) * self.batch_size)
                # Load batch data into memory and prep it
                grids, cells, actions, rewards, next_grids, next_cells = \
                    self.prep_data(
                        h5f['grids'][batch][:],
                        h5f['cells'][batch][:],
                        h5f['chs'][batch][:],
                        h5f['rewards'][batch][:],
                        h5f['next_grids'][batch][:],
                        h5f['next_cells'][batch][:])
                yield {
                    'grids': grids,
                    'cells': cells,
                    'actions': actions,
                    'rewards': rewards,
                    'next_grids': next_grids,
                    'next_cells': next_cells
                }

        train_gen = BackgroundGenerator(data_gen(0, split), k=10)
        test_gen = BackgroundGenerator(data_gen(split, end), k=10)
        return {
            "n_train_steps": split,
            "n_test_steps": end - split,
            "train_gen": train_gen,
            "test_gen": test_gen
        }

    def get_data(self):
        data = np.load("data-experience-shuffle-sub.npy")
        grids, oh_cells, actions, rewards, next_grids, next_oh_cells = \
            self.prep_data(*map(np.array, zip(*data)))

        split_perc = 0.9  # Percentage of data to train on
        split = int(len(grids) * split_perc) // self.batch_size
        end = len(grids) // self.batch_size

        def data_gen(start, stop):
            for i in range(start, stop):
                batch = slice(i * self.batch_size, (i + 1) * self.batch_size)
                yield {
                    'grids': grids[batch],
                    'cells': oh_cells[batch],
                    'actions': actions[batch],
                    'rewards': rewards[batch],
                    'next_grids': next_grids[batch],
                    'next_cells': next_oh_cells[batch]
                }

        train_gen = data_gen(0, split)
        test_gen = data_gen(split, end)
        return {
            "n_train_steps": split,
            "n_test_steps": end - split,
            "train_gen": train_gen,
            "test_gen": test_gen
        }

    @staticmethod
    def prep_data_grids(grids):
        assert type(grids) == np.ndarray
        if grids.ndim == 3:
            grids = np.expand_dims(grids, axis=0)
        grids.shape = (-1, 7, 7, 70)
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
        rewards = rewards.astype(np.float32)  # Needs to be 32-bit

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
            next_data = {
                self.input_grid: data['next_grids'],
                self.input_cell: data['next_cells']
            }
            next_q_maxs = self.sess.run(self.q_max, next_data)
            r = data['rewards']
            q_targets = r + self.gamma * next_q_maxs
            curr_data = {
                self.input_grid: data['grids'],
                self.input_cell: data['cells'],
                self.target_action: data['actions'],
                self.target_q: q_targets
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
                self.input_grid: data['next_grids'],
                self.input_cell: data['next_cells']
            }
            next_q_maxs = self.sess.run(self.q_max, next_data)
            r = data['rewards']
            q_targets = r + self.gamma * next_q_maxs
            curr_data = {
                self.input_grid: data['grids'],
                self.input_cell: data['cells'],
                self.target_action: data['actions'],
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
        q_vals, q_amax, q_max = self.sess.run(
            [self.q_vals, self.q_amax, self.q_max],
            feed_dict={
                self.input_grid: self.prep_data_grids(grid),
                self.input_cell: self.prep_data_cells(cell)
            },
            options=self.options,
            run_metadata=self.run_metadata)
        q_vals = np.reshape(q_vals, [-1])
        return q_vals, q_amax, q_max

    def backward(self, grid, cell, action, q_target):
        data = {
            self.input_grid: self.prep_data_grids(grid),
            self.input_cell: self.prep_data_cells(cell),
            self.target_action: np.array([action], dtype=np.int32),
            self.target_q: np.array([q_target], dtype=np.float32)
        }
        _, loss = self.sess.run(
            [self.do_train, self.loss],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        if np.isnan(loss) or np.isinf(loss):
            self.logger.error(f"Invalid loss: {loss}")
        return loss

    def backward_exp_replay(self, grids, cells, actions, rewards, next_grids,
                            next_cells):
        # TODO Can this be made on-policy, i.e. SARSA? need to store
        # next_actions
        # Get expected returns following a greedy policy from the
        # next state: max a': Q(s', a', w_old)
        next_data = {
            self.input_grid: self.prep_data_grids(next_grids),
            self.input_cell: self.prep_data_cells(next_cells)
        }
        next_q_maxs = self.sess.run(
            self.q_max,
            feed_dict=next_data,
            options=self.options,
            run_metadata=self.run_metadata)
        q_targets = rewards + self.gamma * next_q_maxs
        curr_data = {
            self.input_grid: self.prep_data_grids(grids),
            self.input_cell: self.prep_data_cells(cells),
            self.target_action: actions,
            self.target_q: q_targets
        }
        _, loss = self.sess.run(
            [self.do_train, self.loss],
            feed_dict=curr_data,
            options=self.options,
            run_metadata=self.run_metadata)
        if np.isnan(loss) or np.isinf(loss):
            self.logger.error(f"Invalid loss: {loss}")
        return loss


def scope_vars(scope):
    """
    Get variables inside a scope
    The scope can be specified as a string

    Parameters
    ----------
    scope: str or VariableScope
        scope in which the variables reside.

    Returns
    -------
    vars: [tf.Variable]
        list of variables in `scope`.
    """
    return tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name)


def scope_name():
    """Returns the name of current scope as a string, e.g. deepq/q_func"""
    return tf.get_variable_scope().name


def absolute_scope_name(relative_scope_name):
    """Appends parent scope name to `relative_scope_name`"""
    return scope_name() + "/" + relative_scope_name


if __name__ == "__main__":
    import logging
    logger = logging.getLogger('')
    n = Net(logger)
    n.train()
    # n.eval()
