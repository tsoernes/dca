import numpy as np
import tensorflow as tf


class NeighNet:
    def __init__(self, *args, **kwargs):
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.reset_default_graph()
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.classifier = tf.estimator.Estimator(
            model_fn=self.build_model, model_dir="../model/neighsnet-rhomb")

    def build_model(self, features, labels, mode):
        input_grid = features['input_grid']
        input_cell = features['input_cell']
        input_stacked = tf.concat([input_grid, input_cell], axis=3)
        conv1 = tf.layers.conv2d(
            inputs=input_stacked,
            filters=1,
            kernel_size=5,
            strides=1,
            padding="same",  # pad with 0's
            activation=tf.nn.relu)
        conv1_flat = tf.contrib.layers.flatten(conv1)
        logits = tf.layers.dense(
            inputs=conv1_flat, units=1)
        # Probability of channel not being free for assignment,
        # i.e. it is in use in cell or its neighs2
        inuse = tf.nn.sigmoid(logits, name="sigmoid_tensor")
        self.pred_class = tf.greater(inuse, tf.constant(0.5))
        predictions = {
            "class": tf.greater(inuse, tf.constant(0.5)),
            "probability": inuse
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions)

        loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=labels,
            logits=logits)
        trainer = tf.train.AdamOptimizer(learning_rate=0.0001)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = trainer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(
                    labels=labels,
                    predictions=predictions["class"])}
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    def forward(self, chgrid, cell):
        cg, ce, _ = self.prep_data(chgrid, cell)
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"input_grid": cg, "input_cell": ce},
            shuffle=False)
        preds = self.classifier.predict(input_fn=pred_input_fn)
        prediction = list(p["class"][0] for p in preds)
        return prediction[0]

    def verify_data(self, chgrids, cells, targets):
        from grid import RhombAxGrid
        for i in range(0, len(chgrids)):
            chgrid = chgrids[i]
            cell = cells[i]
            neighs2 = RhombAxGrid.neighbors2(*cell)
            inuse = np.bitwise_or(chgrid[cell], chgrid[neighs2[0]])
            for n in neighs2[1:]:
                inuse = np.bitwise_or(inuse, chgrid[n])
            if not inuse == targets[i]:
                print(i)
                print(chgrid)
                print(chgrid[cell])
                print(cell)
                print(neighs2)
                print("Target: ", targets[i])
                print("Actual: ", inuse)
                raise Exception
        print("\nDATA OK\n")

    def prep_data(self, chgrids, cells, targets=None):
        chgrids = np.array(chgrids)
        oh_cells = np.zeros_like(chgrids)
        # One-hot grid encoding
        for i, cell in enumerate(cells):
            oh_cells[i][cell] = 1
            if cell == (0, 0):
                print(chgrids[i], targets[i])
        chgrids.shape = (-1, 7, 7, 1)
        oh_cells.shape = (-1, 7, 7, 1)
        chgridsneg = chgrids * 2 - 1  # Make empty cells -1 instead of 0
        if targets:
            targets = np.array(targets)
            targets.shape = (-1, 1)
            targets = targets.astype(np.float32)
        chgridsneg = chgridsneg.astype(np.float32)
        oh_cells = oh_cells.astype(np.float32)
        return chgridsneg, oh_cells, targets

    def train(self, do_train=True):
        data = np.load("neighdata-rhomb2.npy")
        # NOTE This data may not be any good, because there's no
        # examples where the ch is free in cell but not free in
        # neighs2
        chgrids, cells, targets = zip(*data)
        chgridsneg, oh_cells, targets = self.prep_data(chgrids, cells, targets)
        # self.verify_data(chgrids, cells, targets)
        split = -10000
        train_data = {"input_grid": chgridsneg[:split],
                      "input_cell": oh_cells[:split]}
        test_data = {"input_grid": chgridsneg[split:],
                     "input_cell": oh_cells[split:]}
        test_targets = targets[split:]

        # Train the model
        if do_train:
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x=train_data,
                y=targets[:split],
                batch_size=100,
                num_epochs=None,
                shuffle=True)
            self.classifier.train(
                input_fn=train_input_fn,
                steps=50000)

        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=test_data,
            y=test_targets,
            num_epochs=1,
            shuffle=False)
        eval_results = self.classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)

        # pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        #   x=test_data,
        #     shuffle=False)
        # preds = self.classifier.predict(input_fn=pred_input_fn)
        # predictions = list(p["class"][0] for p in preds)
        # test_targets.shape = (np.abs(split))
        # wrong_preds = np.where(test_targets != predictions)
        # n_wrong = len(wrong_preds[0])
        # print(f"Number of incorrect preds {n_wrong}"
        #       f"\n Accuracy {(np.abs(split) - n_wrong) / np.abs(split)}")
        # print("The first incorrect examples: ")
        # wrong_grid = test_data['input_grid'][wrong_preds][0]
        # wrong_grid.shape = (7, 7)
        # print(wrong_grid)
        # print(test_data['input_cell'][wrong_preds][0])
        # print(test_targets[wrong_preds][0])
        # print(predictions[wrong_preds[0]])
