import tensorflow as tf

from nets.net import Net
from nets.utils import build_default_trainer, prep_data_grids, scale_freps
from utils import prod


class CACSinghNet(Net):
    def __init__(self, pp, logger, frepshape):
        self.name = "CAC"
        self.frepshape = [None, *frepshape]
        self.grid_inp = pp['singh_grid']
        super().__init__(name=self.name, pp=pp, logger=logger)

    def build(self):
        self.frep = tf.placeholder(tf.int32, self.frepshape, "feature_reps")
        self.reward = tf.placeholder(tf.float32, [], "rewards")
        self.avg_reward = tf.placeholder(tf.float32, [], "avg_reward")
        self.prob_ph = tf.placeholder(tf.float32, [1], "act_prob")
        self.act_ph = tf.placeholder(tf.int32, [1], "act")

        frep = tf.cast(scale_freps(self.frep), tf.float32)
        if self.grid_inp:
            gridshape = [None, self.rows, self.cols, self.n_channels * 2]
            wdim = prod(self.frepshape[1:]) + prod(gridshape[1:])
            self.grid = tf.placeholder(tf.bool, gridshape, "grid")
            grid = tf.cast(self.grid, tf.float32)
            net_inp = tf.concat([grid, frep], axis=3)
        else:
            wdim = prod(self.frepshape[1:])  # Number of parameters in neural net
            net_inp = frep

        net_inp_rv = tf.layers.flatten(net_inp)  # x_t  Row vector
        net_inp_cv = tf.transpose(net_inp_rv)  # Col vector

        hidden = tf.Variable(tf.zeros(shape=(wdim, 1)), name="dense")
        dense = tf.matmul(net_inp_rv, hidden)
        self.prob = tf.nn.sigmoid(dense)
        bernoulli = tf.distributions.Bernoulli(probs=self.prob)
        self.act = bernoulli.sample()

        grads = -(self.reward - self.avg_reward) * (
            tf.cast(self.act_ph, tf.float32) - self.prob_ph) * net_inp_cv

        grads_and_vars = [(grads, hidden)]
        trainer, self.lr, global_step = build_default_trainer(
            net_lr=self.pp['alpha'],
            net_lr_decay=self.pp['alpha_decay'],
            optimizer=self.pp['optimizer'])
        self.do_train = trainer.apply_gradients(grads_and_vars, global_step=global_step)

        return None, None

    def forward(self, freps, grids):
        data = {self.frep: freps}
        if self.grid_inp:
            data[self.grid] = prep_data_grids(grids, self.grid_split)
        p, a = self.sess.run(
            [self.prob, self.act],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        p, a = p[0, 0], a[0, 0]
        # print(a, p)
        return a, p

    def backward(self, *, freps, grids, rewards, avg_reward, actions, action_probs):
        assert len(freps) == 1, (len(freps), type(freps),
                                 freps.shape)  # Hard coded for one-step

        data = {
            self.frep: freps,
            self.reward: rewards,
            self.avg_reward: avg_reward,
            self.act_ph: actions,
            self.prob_ph: action_probs
        }
        if self.grid_inp:
            data[self.grid] = prep_data_grids(grids, self.grid_split)
        _ = self.sess.run(
            [self.do_train],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
