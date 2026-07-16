import numpy as np
from scipy.linalg import solve

class DLinUCB:
    """Discounted Linear UCB (Russac et al. 2019)"""
    def __init__(self, d, ridge_reg=1.0, gamma=0.99, delta=0.05,
                 sigma=1.0, S=1.0, L=1.0):
        self.d = d
        self.ridge_reg = ridge_reg
        self.gamma = gamma
        self.delta = delta
        self.sigma = sigma
        self.S = S
        self.L = L

        self.V = ridge_reg * np.eye(d)
        self.V_tilde = ridge_reg * np.eye(d)
        self.b = np.zeros(d)
        self.theta = np.zeros(d)
        self.t = 0

    def _beta(self):
        t = self.t
        if t == 0:
            term = 0.0
        else:
            gamma2 = self.gamma ** 2
            factor = t if gamma2 == 1 else (1 - gamma2**t) / (1 - gamma2)
            term = self.d * np.log(1 + (self.L**2 * factor) / (self.ridge_reg * self.d))
        return np.sqrt(self.ridge_reg) * self.S + self.sigma * np.sqrt(2 * np.log(1/self.delta) + term)

    def select_action(self, actions):
        beta = self._beta()
        best_val = -np.inf
        best_arm = None
        for a in actions:
            mean = np.dot(a, self.theta)
            z = solve(self.V, a, assume_a='pos')
            std = np.sqrt(np.dot(z, self.V_tilde @ z))
            ucb = mean + beta * std
            if ucb > best_val:
                best_val = ucb
                best_arm = a
        return best_arm

    def update(self, action, reward):
        self.t += 1
        a = action
        self.V = self.gamma * self.V + np.outer(a, a) + (1 - self.gamma) * self.ridge_reg * np.eye(self.d)
        gamma2 = self.gamma ** 2
        self.V_tilde = gamma2 * self.V_tilde + np.outer(a, a) + (1 - gamma2) * self.ridge_reg * np.eye(self.d)
        self.b = self.gamma * self.b + reward * a
        self.theta = solve(self.V, self.b, assume_a='pos')