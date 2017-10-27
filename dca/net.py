import numpy as np
import tensorflow as tf

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
    def __init__(self, restore=False,
                 *args, **kwargs):
        self.alpha = 0.0000001
        self.gamma = 0.9
        self.batch_size = 10

        tf.logging.set_verbosity(tf.logging.INFO)
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.model_path = "model/qnet01/model.cpkt"
        # build or restore
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
        self.input_grid = tf.placeholder(
            shape=[None, 7, 7, 70], dtype=tf.float16, name="input_grid")
        self.input_cell = tf.placeholder(
            shape=[None, 7, 7, 1], dtype=tf.float16, name="input_cell")
        self.target_action = tf.placeholder(
            shape=[None, 1], dtype=tf.int32, name="target_action")
        self.target_q = tf.placeholder(
            shape=[None, 1], dtype=tf.float16, name="target_q")

        input_stacked = tf.concat(
            [self.input_grid, self.input_cell], axis=3)
        conv1 = tf.layers.conv2d(
            inputs=input_stacked,
            filters=70,
            kernel_size=5,
            strides=1,
            padding="same",  # pad with 0's
            activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=70,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu)
        conv2_flat = tf.layers.flatten(conv2)

        # TODO verify that linear neural net performs better than random.
        # Perhaps reducing call rates will increase difference between
        # fixed/random and a good alg, thus making testing nets easier.
        # If so then need to retest sarsa-strats and redo hyperparam opt.
        self.q_vals = tf.layers.dense(inputs=conv2_flat, units=70)
        self.q_amax = tf.argmax(self.q_vals, axis=1)

        flat_q_vals = tf.reshape(self.q_vals, [-1])
        flat_amax = self.q_amax + tf.cast(tf.range(
            tf.shape(self.q_vals)[0]) * tf.shape(self.q_vals)[1], tf.int64)
        self.q_max = tf.gather(flat_q_vals, flat_amax)

        flat_target_action = self.target_action + tf.cast(tf.range(
            tf.shape(self.q_vals)[0]) * tf.shape(self.q_vals)[1], tf.int32)
        predictions = tf.gather(flat_q_vals, flat_target_action)
        # Below we obtain the loss by taking the sum of squares
        # difference between the target and prediction Q values.
        self.loss = tf.losses.mean_squared_error(
            labels=self.target_q,
            predictions=predictions)
        # trainer = tf.train.AdamOptimizer(learning_rate=self.alpha)
        trainer = tf.train.GradientDescentOptimizer(learning_rate=self.alpha)
        # trainer = tf.train.RMSPropOptimizer(learning_rate=self.alpha)
        self.do_train = trainer.minimize(self.loss)

    def train(self):
        data = np.load("data-experience-shuffle.npy")
        grids, cells, actions, rewards, next_grids, next_cells = \
            map(np.array, zip(*data))
        actions = actions.astype(np.int32)
        rewards = rewards.astype(np.float16)

        oh_cells = np.zeros((len(grids), 7, 7), dtype=np.float16)
        next_oh_cells = np.zeros((len(grids), 7, 7), dtype=np.float16)
        # One-hot grid encoding
        for i, cell in enumerate(cells):
            oh_cells[i][cell] = 1
        for i, cell in enumerate(next_cells):
            next_oh_cells[i][cell] = 1
        grids.shape = (-1, 7, 7, 70)
        next_grids.shape = (-1, 7, 7, 70)
        oh_cells.shape = (-1, 7, 7, 1)
        next_oh_cells.shape = (-1, 7, 7, 1)
        actions.shape = (-1, 1)
        rewards.shape = (-1, 1)
        # Make empty cells -1 instead of 0
        grids = grids * 2 - 1
        next_grids = next_grids * 2 - 1

        split_perc = 0.9  # Percentage of data to train on
        split = int(len(grids) * split_perc)
        n_train_steps = split // self.batch_size
        eval_losses = []
        for i in range(len(grids) // self.batch_size):
            # Get expected returns following a greedy policy from the
            # next state: max a': Q(s', a', w_old)
            batch = slice(i * self.batch_size, (i + 1) * self.batch_size)
            next_data = {self.input_grid: next_grids[batch],
                         self.input_cell: next_oh_cells[batch]}
            next_q_maxs = self.sess.run(self.q_max, next_data)
            next_q_maxs.shape = (-1, 1)
            r = rewards[batch]
            q_targets = r + self.gamma * next_q_maxs
            train_data = {self.input_grid: grids[batch],
                          self.input_cell: oh_cells[batch],
                          self.target_action: actions[batch],
                          self.target_q: q_targets}
            if i < n_train_steps:
                loss, _ = self.sess.run([self.loss, self.do_train], train_data)
                print(f"Iter {i} loss: {loss}")
            else:
                if len(eval_losses) == 0:
                    print("Started evaluation")
                eval_losses.append(self.sess.run(self.loss, train_data))
        print(
            f"Trained {i} minibatches of size {self.batch_size}"
            f"\nEval results: {sum(eval_losses) / len(eval_losses)}")


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
