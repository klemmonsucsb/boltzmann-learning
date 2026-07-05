import numpy as np
import pickle
import os
from dlinucb import DLinUCB

class DLinUCBDecisionMaker:
    """Wrapper that matches the DecisionMaker interface used by GamePlay."""
    def __init__(self, action_set, measurement_set,
                 finite_measurements=True,
                 gamma=0.999, lam=1.0, delta=0.05, sigma=1.0,
                 S=1.0, L=1.0, cost_scale=1.0,
                 checkpoint_interval=5000,                 # save every N steps
                 checkpoint_path="dlinucb_checkpoint.pkl",
                 name=None):
        self.name = name if name else self.__class__.__name__
        self.action_set = list(action_set)
        self.measurement_set = list(measurement_set)
        self.finite_measurements = finite_measurements
        self.n_actions = len(action_set)
        self.n_measurements = len(measurement_set)
        self.cost_scale = cost_scale
        self.d = self.n_actions * self.n_measurements

        self.bandit = DLinUCB(d=self.d, lam=lam, gamma=gamma,
                              delta=delta, sigma=sigma, S=S, L=L)

        if self.finite_measurements:
            self.meas_to_idx = {m: i for i, m in enumerate(self.measurement_set)}

        # --- Checkpointing support ---
        self.cost_history = []                 # store the cost of the chosen action
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_path = checkpoint_path

    def _measurement_to_features(self, measurement):
        features = []
        if self.finite_measurements:
            meas_idx = self.meas_to_idx[measurement]
            for a_idx in range(self.n_actions):
                vec = np.zeros(self.d)
                vec[a_idx * self.n_measurements + meas_idx] = 1.0
                features.append(vec)
        else:
            prob_vec = np.array([measurement[m] for m in self.measurement_set])
            for a_idx in range(self.n_actions):
                vec = np.zeros(self.d)
                start = a_idx * self.n_measurements
                vec[start:start+self.n_measurements] = prob_vec
                features.append(vec)
        return features

    def get_action(self, measurement, time=0.0, **kwargs):
        features = self._measurement_to_features(measurement)
        best_vec = self.bandit.select_action(features)
        for a, vec in zip(self.action_set, features):
            if np.array_equal(best_vec, vec):
                return a, None, None
        return self.action_set[0], None, None

    def update_energies(self, measurement, costs, time=0.0, **kwargs):
        action = kwargs.get('action')
        if action is None:
            raise ValueError("DLinUCBDecisionMaker.update_energies needs 'action' kwarg.")
        features = self._measurement_to_features(measurement)
        action_idx = self.action_set.index(action)
        feat = features[action_idx]
        reward = -costs[action] / self.cost_scale
        self.bandit.update(feat, reward)

        # Record the actual cost incurred by the chosen action
        self.cost_history.append(costs[action])

        # Save a checkpoint periodically
        if len(self.cost_history) % self.checkpoint_interval == 0:
            self.save_checkpoint()

    def save_checkpoint(self):
        """Save partial results and the full bandit state."""
        data = {
            'cost_history': self.cost_history,
            'bandit_state': {
                'V': self.bandit.V,
                'V_tilde': self.bandit.V_tilde,
                'b': self.bandit.b,
                'theta': self.bandit.theta,
                't': self.bandit.t,
                'gamma': self.bandit.gamma,
                'lam': self.bandit.lam,
            }
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Checkpoint saved at step {len(self.cost_history)}")

    def load_checkpoint(self):
        """Restore from a checkpoint. Returns True if successful."""
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            self.cost_history = data['cost_history']
            state = data['bandit_state']
            self.bandit.V = state['V']
            self.bandit.V_tilde = state['V_tilde']
            self.bandit.b = state['b']
            self.bandit.theta = state['theta']
            self.bandit.t = state['t']
            self.bandit.gamma = state['gamma']
            self.bandit.lam = state['lam']
            return True
        return False

    def reset(self):
        self.bandit = DLinUCB(d=self.d, lam=self.bandit.lam,
                              gamma=self.bandit.gamma,
                              delta=self.bandit.delta,
                              sigma=self.bandit.sigma,
                              S=self.bandit.S, L=self.bandit.L)
        self.cost_history = []

