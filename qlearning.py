import numpy as np
from collections import defaultdict

class QLearning:
    def __init__(self, learning_rate, discount_factor, n_actions):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_discount_factor(self, discount_factor):
        self.discount_factor = discount_factor

    def get_q(self, state):
        return self.q_table[state]

    def get_max_q(self, state):
        return np.max(self.q_table[state])

    def update_q_table(self, state, new_state, action, reward, done, goal_condition):

        max_future_q = np.max(self.q_table[new_state])
        current_q = self.q_table[state][action]
        temporal_difference = reward + self.discount_factor * max_future_q - current_q
        new_q = current_q + self.learning_rate * temporal_difference

        if done and goal_condition:
            self.q_table[state][action] = reward
        else:
            self.q_table[state][action] = new_q
