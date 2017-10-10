import numpy as np


class SARSA:
    def __init__(self):
        self.epsilon = 1
        self.gamma = 1
        self.net = None
        self.qvals = np.zeros(0)

    def sim(self):
        state = None
        action = self.get_init_action(state)
        while True:
            next_state, reward = self.execute_action(state, action)
            next_action = self.get_action(
                    next_state, reward, state, action)
            state = next_state
            action = next_action

    def get_action(self, next_state, reward, state, action):
        # Epsilon-greedy choose action
        valid_actions = None
        next_action, next_qvals = self.net.forward(next_state)
        if np.random.random() < self.epsilon:
            next_action = np.random.choice(valid_actions)

        # Update q-values
        td_err = reward + self.gamma * (next_qvals[next_action])
        q_target = td_err - self.qvals[action]
        td_errs = np.zeros(len(self.qvals))
        td_errs[action] = q_target
        self.net.backward(td_errs)
        self.qvals = next_qvals
        return action
