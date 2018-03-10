import numpy as np

from nets.singh_man import ManSinghNet


class ResidSinghNet(ManSinghNet):
    def __init__(self, *args, **kwargs):
        """
        Residual gradient
        """
        super().__init__(*args, **kwargs)

    def backward(self, freps, rewards, next_freps, discount=None):
        value = self.sess.run(self.value, feed_dict={self.freps: freps})[0][0]
        next_value = self.sess.run(self.value, feed_dict={self.freps: next_freps})[0][0]
        td_err = rewards[0] + discount * next_value - value
        frep_colvec = np.reshape(freps[0], [-1, 1])
        next_frep_colvec = np.reshape(next_freps[0], [-1, 1])
        # Residual gradient
        grad = -2 * td_err * (frep_colvec - discount * next_frep_colvec)
        data = {self.freps: freps, self.grads[0][0]: grad}
        lr, _ = self.sess.run([self.lr, self.do_train], data)
        assert not np.isnan(td_err) or not np.isinf(td_err)
        return td_err**2, lr, td_err
