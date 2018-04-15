import numpy as np

from nets.singh_man import ManSinghNet


class TDLSinghNet(ManSinghNet):
    def __init__(self, pp, logger, frepshape):
        """
        True online td lambda
        """
        self.frepshape = frepshape
        super().__init__(pp, logger, frepshape)
        self.z = np.zeros((self.wdim, 1))
        self.lmbda = self.pp['lambda']
        assert self.lmbda is not None
        assert pp['target'] == 'discount'
        self.v_old = 0

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
        assert len(freps) == 1  # Hard coded for one-step
        value, next_value, inp_colvec, next_inp_colvec, lr = self._get_vals_inps(
            freps, next_freps, grids, next_grids)

        td_err = rewards[0] + discount * next_value - value

        dot = np.dot(inp_colvec.T, self.z)
        self.z = discount * self.lmbda * self.z + (
            1 - lr * discount * self.lmbda * dot) * inp_colvec
        z_colvec = np.reshape(self.z, [-1, 1])
        grad = (td_err + value - self.v_old) * z_colvec - (
            value - self.v_old) * inp_colvec
        data = {self.grads[0][0]: -grad}
        lr, _ = self.sess.run(
            [self.lr, self.do_train],
            feed_dict=data,
            options=self.options,
            run_metadata=self.run_metadata)
        self.v_old = next_value
        return td_err**2, lr, td_err
