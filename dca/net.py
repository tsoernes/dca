from utils import BackgroundGenerator

import h5py
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

"""
Neighbors2
(3,3)
min row: 1
max row: 5
min col: 1
max col: 5
(4,3)
min row: 2
max row: 6
min col: 1
max col: 5
So it might be a good idea to have 4x4 filters,
as that would cover all neighs2

Padding with 0's is the natural choice since that would be
equivalent to having empty cells outside of grid

For a policy network, i.e. with actions [0, 1, ..., n_channels-1]
corresponding to the probability of assigning the different channels,
how can the network know, or be trained to know, that some actions
are illegal/unavailable?
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
start very small (eg 1e-6), make sure it's barely changing
or decreasing very slowly.
If cost is NaN or inf, learning rate is too high

on tuning hyperparams:
if cost goes over 3x original cost, break out early

big gap between train and test accuracy:
overfitting. reduce net size or increase regularization
no gap: increase net size

debugging nets: track ratio of weight updates/weight magnitues
should be somewhere around 0.001 or so. if too high, decrease
learning rate, if too log (like 1e-6), increase lr.
"""


class Net:
    def __init__(self, restore=True, save=True,
                 *args, **kwargs):
        self.save = save
        self.alpha = 1e-7
        self.gamma = 0.9
        self.batch_size = 10
        self.model_path = "model/qnet01/model.cpkt"
        self.log_path = "model/qnet01/logs"

        tf.logging.set_verbosity(tf.logging.INFO)
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.build()
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init)
        if restore:
            # Could do a try/except and build if loading fails
            print(f"Restoring model from {self.model_path}")
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

    def build(self):
        self.input_grid = tf.placeholder(
            shape=[None, 7, 7, 70], dtype=tf.float32, name="input_grid")
        self.input_cell = tf.placeholder(
            shape=[None, 7, 7, 1], dtype=tf.float32, name="input_cell")
        self.target_action = tf.placeholder(
            shape=[None], dtype=tf.int32, name="target_action")
        self.target_q = tf.placeholder(
            shape=[None], dtype=tf.float32, name="target_q")

        # NOTE Could stack after conv. No need to do conv on oh cells
        input_stacked = tf.concat(
            [self.input_grid, self.input_cell], axis=3)
        conv1 = tf.layers.conv2d(
            inputs=input_stacked,
            filters=70,
            kernel_size=5,
            padding="same",
            activation=tf.nn.relu)
        # Conv2d_in_plane does not support float16
        # conv1 = tf.contrib.layers.conv2d_in_plane(
        #     inputs=input_stacked,
        #     kernel_size=5,
        #     stride=1,
        #     padding="SAME",  pad with 0's
        #     activation_fn=tf.nn.relu)
        # conv2 = tf.layers.conv2d(
        #     inputs=conv1,
        #     filters=140,
        #     kernel_size=3,
        #     padding="same",
        #     activation=tf.nn.relu)
        conv2_flat = tf.layers.flatten(conv1)

        # Perhaps reducing call rates will increase difference between
        # fixed/random and a good alg, thus making testing nets easier.
        # If so then need to retest sarsa-strats and redo hyperparam opt.
        dense = tf.layers.dense(
            inputs=conv2_flat, units=128, name="dense")
        self.q_vals = tf.layers.dense(
            inputs=conv2_flat, units=70, name="q_vals")
        self.q_amax = tf.argmax(self.q_vals, axis=1, name="q_amax")

        flat_q_vals = tf.reshape(self.q_vals, [-1])
        flat_amax = self.q_amax + tf.cast(tf.range(
            tf.shape(self.q_vals)[0]) * tf.shape(self.q_vals)[1], tf.int64)
        self.q_max = tf.gather(flat_q_vals, flat_amax)

        flat_target_action = self.target_action + tf.cast(tf.range(
            tf.shape(self.q_vals)[0]) * tf.shape(self.q_vals)[1], tf.int32)
        self.predictions = tf.gather(
            flat_q_vals, flat_target_action, name="action_q_vals")
        # Below we obtain the loss by taking the sum of squares
        # difference between the target and prediction Q values.
        self.loss = tf.losses.mean_squared_error(
            labels=self.target_q,
            predictions=self.predictions)
        trainer = tf.train.AdamOptimizer(
            learning_rate=self.alpha)
        # trainer = tf.train.GradientDescentOptimizer(learning_rate=self.alpha)
        # trainer = tf.train.RMSPropOptimizer(learning_rate=self.alpha)
        self.do_train = trainer.minimize(self.loss)
        with tf.name_scope("summaries"):
            tf.summary.scalar("learning_rate", self.alpha)
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("qvals", self.q_vals)
        self.summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(
            self.log_path + '/train', self.sess.graph)
        self.eval_writer = tf.summary.FileWriter(
            self.log_path + '/eval')

        self.shapes = \
            [tf.shape(input_stacked),
             tf.shape(conv1),
             tf.shape(conv2_flat),
             tf.shape(dense),
             tf.shape(self.q_vals),
             tf.shape(self.q_amax),
             tf.shape(flat_q_vals),
             tf.shape(flat_amax),
             tf.shape(self.q_max),
             tf.shape(flat_target_action),
             tf.shape(self.predictions),
             tf.shape(self.loss)]

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
                    'next_cells': next_cells}

        train_gen = BackgroundGenerator(data_gen(0, split), k=10)
        test_gen = BackgroundGenerator(data_gen(split, end), k=10)
        return {"n_train_steps": split, "n_test_steps": end - split,
                "train_gen": train_gen, "test_gen": test_gen}

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
                    'next_cells': next_oh_cells[batch]}

        train_gen = data_gen(0, split)
        test_gen = data_gen(split, end)
        return {"n_train_steps": split, "n_test_steps": end - split,
                "train_gen": train_gen, "test_gen": test_gen}

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
        print(f"Training {self.n_train_steps} minibatches of size"
              f" {self.batch_size} for a total of"
              f" {self.n_train_steps * self.batch_size} examples")
        for i in range(self.n_train_steps):
            # Get expected returns following a greedy policy from the
            # next state: max a': Q(s', a', w_old)
            data = next(self.train_gen)
            next_data = {self.input_grid: data['next_grids'],
                         self.input_cell: data['next_cells']}
            next_q_maxs = self.sess.run(self.q_max, next_data)
            r = data['rewards']
            q_targets = r + self.gamma * next_q_maxs
            curr_data = {
                self.input_grid: data['grids'],
                self.input_cell: data['cells'],
                self.target_action: data['actions'],
                self.target_q: q_targets}
            _, loss, summary = self.sess.run(
                [self.do_train, self.loss, self.summaries],
                curr_data)
            if i % 50 == 0:
                # tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # run_metadata = tf.RunMetadata()
                # self.train_writer.add_run_metadata(
                #     run_metadata, 'step%d' % i)
                self.train_writer.add_summary(summary, i)
                print(f"Iter {i}\tloss: {loss:.2f}")
                losses.append(loss)
            if np.isnan(loss) or np.isinf(loss):
                print(f"Invalid loss: {loss}")
                break
        if self.save:
            print(f"Saving model to path {self.model_path}")
            self.saver.save(self.sess, self.model_path)
        self.eval()

        plt.plot(losses)
        plt.ylabel("Loss")
        plt.xlabel(f"Iterations, in {self.batch_size}s")
        plt.show()

    def eval(self):
        self.load_data()
        print(f"Evaluating {self.n_test_steps} minibatches of size"
              f" {self.batch_size} for a total of"
              f"  {self.n_test_steps * self.batch_size} examples")
        eval_losses = []
        for i in range(self.n_test_steps):
            # Get expected returns following a greedy policy from the
            # next state: max a': Q(s', a', w_old)
            data = next(self.test_gen)
            next_data = {self.input_grid: data['next_grids'],
                         self.input_cell: data['next_cells']}
            next_q_maxs = self.sess.run(self.q_max, next_data)
            r = data['rewards']
            q_targets = r + self.gamma * next_q_maxs
            curr_data = {
                self.input_grid: data['grids'],
                self.input_cell: data['cells'],
                self.target_action: data['actions'],
                self.target_q: q_targets}
            loss, summary = self.sess.run(
                [self.loss, self.summaries],
                curr_data)
            self.eval_writer.add_summary(summary, i)
            eval_losses.append(loss)
            if np.isnan(loss) or np.isinf(loss):
                print(f"Invalid loss: {loss}")
                break
        print(f"\nEval results: {sum(eval_losses) / len(eval_losses)}")

    def forward(self, grid, cell):
        q_vals, q_amax, q_max = self.sess.run(
            [self.q_vals, self.q_amax, self.q_max],
            feed_dict={
                self.input_grid: self.prep_data_grids(grid),
                self.input_cell: self.prep_data_cells(cell)})
        q_vals = np.reshape(q_vals, [-1])
        return q_vals, q_amax, q_max

    def backward(self, grid, cell, action, q_target):
        data = {
            self.input_grid: self.prep_data_grids(grid),
            self.input_cell: self.prep_data_cells(cell),
            self.target_action: np.array([action], dtype=np.int32),
            self.target_q: np.array([q_target], dtype=np.float32)}
        self.sess.run(
            self.loss,
            data)


class FreeChNet:
    def __init__(self, restore=False, *args, **kwargs):
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.model_path = "model/freechnet/model.cpkt"
        # build OR restore
        if restore:
            init = tf.global_variables_initializer()
            self.sess.run(init)
            print(f"Restoring model from {self.model_path}")
            self.saver.restore(self.sess, self.model_path)
        else:
            self.build()
            init = tf.global_variables_initializer()
            self.sess.run(init)
        self.saver = tf.train.Saver()

    def build(self):
        # Result: ~0.95 accuracy after 25 steps with batch size of 100
        # and learning rate of 0.001 (Momentum)
        # Should start with higher learning rate, 0.01 and decay
        self.tfinput_grid = tf.placeholder(
            shape=[None, 7, 7, 70], dtype=tf.float16, name="input_grid")

        # TODO Test -1 for empty entries on input_cell
        self.tfinput_cell = tf.placeholder(
            shape=[None, 7, 7, 1], dtype=tf.float16, name="input_cell")
        self.tflabels = tf.placeholder(
            shape=[None, 70], dtype=tf.float16, name="labels")
        input_stacked = tf.concat(
            [self.tfinput_grid, self.tfinput_cell], axis=3)
        conv1 = tf.layers.conv2d(
            inputs=input_stacked,
            filters=70,
            kernel_size=5,
            strides=1,
            padding="same",  # pad with 0's
            activation=tf.nn.relu)
        # conv2 = tf.layers.conv2d(
        #     inputs=conv1,
        #     filters=70,
        #     kernel_size=1,
        #     strides=1,
        #     padding="same",
        #     activation=tf.nn.relu)
        conv_flat = tf.contrib.layers.flatten(conv1)
        logits = tf.layers.dense(inputs=conv_flat, units=70)
        prob_inuse = tf.add(
            tf.nn.sigmoid(logits, name="sigmoid_tensor"), 0.000000000001)
        self.inuse = tf.greater(prob_inuse, tf.constant(0.5, dtype=tf.float16))

        # TODO How does this work then there's multiple batches?
        # Should do a manual step through and look
        self.loss = tf.losses.sigmoid_cross_entropy(
            self.tflabels,
            logits=logits)
        # trainer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        trainer = tf.train.MomentumOptimizer(
            momentum=0.9,
            learning_rate=0.001)
        self.updateModel = trainer.minimize(self.loss)

    def predict(self, grids, cells):
        # If there's no outer array for batches, the reshaping will mangle
        # the data
        assert len(grids.shape) == 4
        assert len(cells.shape) == 4
        pgrids, pcells, _ = self.prep_data(grids, cells)
        predictions = self.sess.run(
            self.inuse,
            {self.tfinput_grid: pgrids,
             self.tfinput_cell: pcells})
        return predictions

    def train(self, do_train=True):
        data = np.load("data-freechs-shuffle.npy")
        # data = data[:50000]
        grids, cells, targets = zip(*data)
        grids, oh_cells, targets = self.prep_data(grids, cells, targets)
        split = -1000
        train_grids = grids[:split]
        train_cells = oh_cells[:split]
        train_targets = targets[:split]
        test_grids = grids[split:]
        test_cells = oh_cells[split:]
        test_targets = targets[split:]

        # Train the model
        batch_size = 100
        invalid = False
        # steps = len(train_grids) // batch_size
        steps = 20
        for i in range(steps):
            bgrids = train_grids[i * batch_size: (i + 1) * batch_size]
            bcells = train_cells[i * batch_size: (i + 1) * batch_size]
            btargets = train_targets[i * batch_size: (i + 1) * batch_size]
            _, loss, preds = self.sess.run(
                [self.updateModel, self.loss, self.inuse],
                {self.tfinput_grid: bgrids,
                 self.tfinput_cell: bcells,
                 self.tflabels: btargets})
            if np.isinf(loss) or np.isnan(loss):
                print(f"Invalid loss iter {i}: {loss}")
                invalid = True
            if (i % 1 == 0):
                # Might do this in tf instead
                accuracy = np.sum(preds == btargets) / preds.size
                nonzero = np.sum(np.any(preds, axis=1))
                print(
                    f"\nLoss at step {i}: {loss}"
                    f"\nMinibatch accuracy: {accuracy:.4f}%"
                    "\nNumber of not all-zero predictions:"
                    f" {nonzero}/{len(preds)}")
        print(f"Finished training. Encountered invalid loss? {invalid}")
        self.saver.save(self.sess, self.model_path)
        print(f"Saving model to {self.model_path}")

        self.eval(test_grids, test_cells, test_targets)

    def eval(self, grids, cells, targets):
        batch_size = 50
        losses, accuracies, aon_accuracies = [], [], []
        for i in range(len(grids) // batch_size):
            bgrids = grids[i * batch_size: (i + 1) * batch_size]
            bcells = cells[i * batch_size: (i + 1) * batch_size]
            btargets = targets[i * batch_size: (i + 1) * batch_size]
            loss, preds = self.sess.run(
                [self.loss, self.inuse],
                {self.tfinput_grid: bgrids,
                 self.tfinput_cell: bcells,
                 self.tflabels: btargets})
            losses.append(loss)
            accuracy = np.sum(preds == btargets) / np.size(preds)
            accuracies.append(accuracy)
            aon_accuracy = \
                np.sum(np.all((preds == btargets), axis=1)) / len(preds)
            aon_accuracies.append(aon_accuracy)
        print(f"Evaluating {len(grids)} samples")
        print(f"Loss: {sum(losses)/len(losses)}")
        print(f"Accuracy: {sum(accuracies)/len(accuracies):.8f}%")
        print("All-or-nothing accuracy:"
              f" {sum(aon_accuracies)/len(aon_accuracies):.8f}%")
        n = 3
        i, miss, hit = 0, 0, 0
        while (miss < n) and (hit < n):
            bgrids = grids[i:i + 1]
            bcells = cells[i:i + 1]
            btargets = targets[i:i + 1]
            _, preds = self.sess.run(
                [self.loss, self.inuse],
                {self.tfinput_grid: bgrids,
                 self.tfinput_cell: bcells,
                 self.tflabels: btargets})
            idx = np.where(bcells[0])
            cell = (idx[0][0], idx[1][0])
            if not (preds == btargets).all():
                miss += 1
                print(f"\nMisclassified example #{i}\nCell: {cell}")
            else:
                hit += 1
                print(f"\nCorrectly classified example #{i}\nCell: {cell}")
            print(f"Targets: {btargets[0].astype(int)}")
            print(f"Preds:\t{preds[0].astype(int)}")
            i += 1

    def prep_data(self, grids, cells, targets=None):
        grids = np.array(grids)
        oh_cells = np.zeros((len(grids), 7, 7), dtype=np.float16)
        # One-hot grid encoding
        for i, cell in enumerate(cells):
            oh_cells[i][cell] = 1
        if targets is not None:
            h_targets = np.zeros((len(grids), 70), dtype=np.float16)
            for i, targ in enumerate(targets):
                for ch in targ:
                    h_targets[i][ch] = 1
        grids.shape = (-1, 7, 7, 70)
        oh_cells.shape = (-1, 7, 7, 1)
        gridsneg = grids * 2 - 1  # Make empty cells -1 instead of 0
        return gridsneg, oh_cells, h_targets


if __name__ == "__main__":
    n = Net()
    n.train()
    # n.eval()
