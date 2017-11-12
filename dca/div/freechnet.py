import numpy as np
import tensorflow as tf


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
            self.tflabels, logits=logits)
        # trainer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        trainer = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=0.001)
        self.updateModel = trainer.minimize(self.loss)

    def predict(self, grids, cells):
        # If there's no outer array for batches, the reshaping will mangle
        # the data
        assert len(grids.shape) == 4
        assert len(cells.shape) == 4
        pgrids, pcells, _ = self.prep_data(grids, cells)
        predictions = self.sess.run(
            self.inuse, {self.tfinput_grid: pgrids,
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
            bgrids = train_grids[i * batch_size:(i + 1) * batch_size]
            bcells = train_cells[i * batch_size:(i + 1) * batch_size]
            btargets = train_targets[i * batch_size:(i + 1) * batch_size]
            _, loss, preds = self.sess.run(
                [self.updateModel, self.loss, self.inuse], {
                    self.tfinput_grid: bgrids,
                    self.tfinput_cell: bcells,
                    self.tflabels: btargets
                })
            if np.isinf(loss) or np.isnan(loss):
                print(f"Invalid loss iter {i}: {loss}")
                invalid = True
            if (i % 1 == 0):
                # Might do this in tf instead
                accuracy = np.sum(preds == btargets) / preds.size
                nonzero = np.sum(np.any(preds, axis=1))
                print(f"\nLoss at step {i}: {loss}"
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
            bgrids = grids[i * batch_size:(i + 1) * batch_size]
            bcells = cells[i * batch_size:(i + 1) * batch_size]
            btargets = targets[i * batch_size:(i + 1) * batch_size]
            loss, preds = self.sess.run([self.loss, self.inuse], {
                self.tfinput_grid: bgrids,
                self.tfinput_cell: bcells,
                self.tflabels: btargets
            })
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
            _, preds = self.sess.run([self.loss, self.inuse], {
                self.tfinput_grid: bgrids,
                self.tfinput_cell: bcells,
                self.tflabels: btargets
            })
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
