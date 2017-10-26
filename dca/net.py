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
    def __init__(self,
                 *args, **kwargs):
        self.alpha = 0.01
        self.gamma = 0.9
        self.batch_size = 100
        tf.logging.set_verbosity(tf.logging.INFO)
        # tf.reset_default_graph()
        # self.sess = tf.Session()
        # init = tf.global_variables_initializer()
        # self.sess.run(init)
        self.classifier = tf.estimator.Estimator(
            model_fn=self.build_model, model_dir="../model/qnet01")

    def build_model(self, features, labels, mode):
        input_grid = features['grid']
        input_cell = features['cell']
        input_stacked = tf.concat([input_grid, input_cell], axis=3)
        # self.tfinput_grid = tf.placeholder(
        #      shape=[None, 7, 7, 70], dtype=tf.float16, name="input_grid")
        # self.tfinput_cell = tf.placeholder(
        #     shape=[None, 7, 7, 1], dtype=tf.float16, name="input_cell")
        # input_stacked = tf.concat(
        #     [self.tfinput_grid, self.tfinput_cell], axis=3)
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
        self.q_max = tf.maximum(self.q_vals, axis=1)

        # Below we obtain the loss by taking the sum of squares
        # difference between the target and prediction Q values.
        # self.targets = tf.placeholder(
        #     shape=[1, 70], dtype=tf.float32, name="targets")

        predictions = {
            "qvals": self.q_vals,
            "q_max": self.q_max,
            "q_amax": self.q_amax,
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions)

        loss = tf.losses.mean_squared_error(
            labels=labels['q_max'],
            predictions=self.q_vals[labels['actions']])
        trainer = tf.train.AdamOptimizer(learning_rate=self.alpha)
        # trainer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
        # trainer = tf.train.RMSPropOptimizer(learning_rate=alpha)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = trainer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {
                "accuracy": tf.metrics.mean_squared_error(
                    labels=labels['q_max'],
                    predictions=self.q_vals[labels['actions']])}
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    def get_data(self, fname):
        data = np.load(fname)
        # grids = np.array(data[:, 0])
        # cells = np.array(data[:, 1])
        # actions = np.array(data[:, 2])
        # rewards = np.array(data[:, 3])
        # next_grids = np.array(data[:, 4])
        # next_cells = np.array(data[:, 5])
        grids, cells, actions, rewards, next_grids, next_cells = \
            map(np.array, zip(*data))

        oh_cells = np.zeros((len(grids), 7, 7), dtype=np.float16)
        next_oh_cells = np.zeros((len(grids), 7, 7), dtype=np.float16)
        # One-hot grid encoding
        for i, cell in enumerate(cells):
            oh_cells[i][cell] = 1
        for i, cell in enumerate(next_cells):
            next_oh_cells[i][cell] = 1
        grids.shape = (-1, 7, 7, 70)  # should this be -1,7,7,70,1 perhaps
        next_grids.shape = (-1, 7, 7, 70)
        oh_cells.shape = (-1, 7, 7, 1)
        next_oh_cells.shape = (-1, 7, 7, 1)
        grids = grids * 2 - 1  # Make empty cells -1 instead of 0
        next_grids = next_grids * 2 - 1

        # TODO WHAT TO DO WITH ACTIONS
        split = -1000
        train_data = {"grid": grids[:split],
                      "cell": oh_cells[:split],
                      "actions": actions[:split]}
        test_data = {"grid": grids[split:],
                     "cell": oh_cells[split:],
                     "actions": actions[split:]}
        # Get expected returns following a greedy policy from the next state
        # max a': Q(s', a', w_old)
        nextq_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"grid": next_grids, "cells": next_cells},
            batch_size=self.batch_size,
            num_epochs=None,
            shuffle=False)  # Data already shuffled
        next_max_qs = self.classifier.predict(
            input_fn=nextq_input_fn,
            predict_keys=['q_max'])
        targets = rewards + self.gamma * next_max_qs
        test_targets = targets[:split]
        train_targets = targets[split:]
        return train_data, train_targets, test_data, test_targets

    def train(self):
        data = self.get_data("data-experience-shuffle.npy")
        train_data, train_targets, test_data, test_targets = data

        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=train_data,
            y=train_targets,
            batch_size=self.batch_size,
            num_epochs=None,
            shuffle=False)  # Data already shuffled
        self.classifier.train(
            input_fn=train_input_fn,
            steps=5000)

        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=test_data,
            y=test_targets,
            num_epochs=1,
            shuffle=False)
        eval_results = self.classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)


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
