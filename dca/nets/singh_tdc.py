import numpy as np

from nets.singh_man import ManSinghNet


class TDCSinghNet(ManSinghNet):
    def __init__(self, pp, logger, frepshape):
        """
        TD0 with Gradient correction
        """
        super().__init__(pp, logger, frepshape)
        self.grad_beta = self.pp['grad_beta']
        self.grad_beta_decay = 1 - self.pp['grad_beta_decay']
        # Weights is a column vector of same length as flatten net input
        self.weights = np.zeros((self.wdim, 1))

    def backward(self,
                 *,
                 freps,
                 rewards,
                 next_freps,
                 discount=None,
                 weights=[1],
                 avg_reward=None,
                 grids=None,
                 next_grids=None,
                 **kwargs):
        # NOTE can possible take in val, next_val here as theyre already known
        assert len(freps) == 1  # Hard coded for one-step
        assert weights is not None
        assert weights[0] is not None
        value, next_value, inp_colvec, next_inp_colvec, _ = self._get_vals_inps(
            freps, next_freps, grids, next_grids)

        if avg_reward is None:
            avg_reward = 0
        else:
            discount = 1
        td_err = rewards[0] + discount * next_value - value
        # dot is inner product and therefore a scalar
        dot = np.dot(inp_colvec.T, self.weights)
        grad = -2 * weights[0] * (
            td_err * inp_colvec + avg_reward - discount * next_inp_colvec * dot)
        lr, _ = self.sess.run(
            [self.lr, self.do_train],
            feed_dict={self.grads[0][0]: grad},
            options=self.options,
            run_metadata=self.run_metadata)
        self.weights += self.grad_beta * (td_err - dot) * inp_colvec
        self.grad_beta *= self.grad_beta_decay
        return td_err**2, lr, td_err
