import numpy as np
import pickle
import os
from dlinucb import DLinUCB


class DLinUCBDecisionMaker:
    """Wrapper that matches the DecisionMaker interface used by GamePlay.

    Key design choice: for the finite-measurement setting, we maintain one small
    DLinUCB bandit *per measurement/context*, each of dimension n_actions, rather
    than a single DLinUCB over the full (n_actions * n_measurements)-dimensional
    one-hot space.

    Why this matters: DLinUCB's exploration bonus (beta_t in the paper) grows with
    the ambient dimension d (see Eq. 5 of Russac et al. -- there's a `d * log(...)`
    term inside the sqrt). Because our features are one-hot over (action, measurement)
    pairs, the design matrix V is block-diagonal: each context's block is only ever
    updated by observations from that context. So the *point estimate* theta_hat
    would end up numerically similar whether we use one big bandit or many small
    ones. But beta_t is a single global scalar sized for the full d=729 case, so a
    single big bandit pays a d=729-sized exploration bonus on every single action,
    even though each context only ever lives in a 3-dimensional slice. That bonus
    barely shrinks over 100k rounds spread across 243 contexts, so the algorithm
    stays in "explore" mode near-permanently and never commits to good actions.
    Splitting into per-context 3-dim bandits gives each context its own
    appropriately-sized beta_t that shrinks quickly once that context has been
    seen a reasonable number of times.
    """

    def __init__(self, action_set, measurement_set,
                 finite_measurements=True,
                 gamma=0.999, ridge_reg=1.0, delta=0.05, sigma=1.0,
                 S=1.0, L=1.0, cost_scale=1.0,
                 checkpoint_interval=5000,                 # save every N steps
                 checkpoint_path=None,
                 name=None):
        self.name = name if name else self.__class__.__name__
        self.action_set = list(action_set)
        self.measurement_set = list(measurement_set)
        self.finite_measurements = finite_measurements
        self.n_actions = len(action_set)
        self.n_measurements = len(measurement_set)
        self.cost_scale = cost_scale

        self._gamma = gamma
        self._ridge_reg = ridge_reg
        self._delta = delta
        self._sigma = sigma
        self._S = S
        self._L = L

        if self.finite_measurements:
            # per-context bandit: each one only ever sees a 3-dim one-hot action space
            self.d = self.n_actions
            self.meas_to_idx = {m: i for i, m in enumerate(self.measurement_set)}
            self._bandits = {}  # lazily created, one per measurement
        else:
            # continuum measurements can't be split into discrete per-context bandits,
            # so we fall back to a single bandit over the full feature space.
            self.d = self.n_actions * self.n_measurements
            self.bandit = DLinUCB(d=self.d, ridge_reg=ridge_reg, gamma=gamma,
                                  delta=delta, sigma=sigma, S=S, L=L)

        # --- Checkpointing support ---
        self.cost_history = []                 # store the cost of the chosen action
        self.checkpoint_interval = checkpoint_interval
        # FIX: namespace the checkpoint file by `name` so that sweeping multiple
        # DLinUCBDecisionMaker instances (e.g. one per lambda) doesn't have them
        # all silently overwrite the same "dlinucb_checkpoint.pkl" mid-run.
        if checkpoint_path is None:
            safe_name = self.name.replace('/', '_').replace(' ', '_')
            checkpoint_path = f"dlinucb_checkpoint_{safe_name}.pkl"
        self.checkpoint_path = checkpoint_path

    def _get_agent(self, measurement):
        """Lazily create/retrieve the per-context DLinUCB bandit for finite measurements."""
        if measurement not in self._bandits:
            self._bandits[measurement] = DLinUCB(d=self.d, ridge_reg=self._ridge_reg, gamma=self._gamma,
                                                 delta=self._delta, sigma=self._sigma,
                                                 S=self._S, L=self._L)
        return self._bandits[measurement]

    def _measurement_to_features(self, measurement):
        """Build the per-action feature vectors for the *current* bandit's dimension."""
        features = []
        if self.finite_measurements:
            for a_idx in range(self.n_actions):
                vec = np.zeros(self.d)
                vec[a_idx] = 1.0
                features.append(vec)
        else:
            prob_vec = np.array([measurement[m] for m in self.measurement_set])
            for a_idx in range(self.n_actions):
                vec = np.zeros(self.d)
                start = a_idx * self.n_measurements
                vec[start:start + self.n_measurements] = prob_vec
                features.append(vec)
        return features

    def get_action(self, measurement, time=0.0, **kwargs):
        features = self._measurement_to_features(measurement)
        bandit = self._get_agent(measurement) if self.finite_measurements else self.bandit
        best_vec = bandit.select_action(features)
        for a, vec in zip(self.action_set, features):
            if np.array_equal(best_vec, vec):
                return a, None, None
        return self.action_set[0], None, None

    def update_energies(self, measurement, costs, time=0.0, **kwargs):
        action = kwargs.get('action')
        if action is None:
            raise ValueError("DLinUCBDecisionMaker.update_energies needs 'action' kwarg.")
        features = self._measurement_to_features(measurement)
        bandit = self._get_agent(measurement) if self.finite_measurements else self.bandit
        action_idx = self.action_set.index(action)
        feat = features[action_idx]
        reward = -costs[action] / self.cost_scale
        bandit.update(feat, reward)

        # Record the actual cost incurred by the chosen action
        self.cost_history.append(costs[action])

        # Save a checkpoint periodically
        if len(self.cost_history) % self.checkpoint_interval == 0:
            self.save_checkpoint()

    def save_checkpoint(self):
        """Save partial results and the full bandit state (all per-context bandits)."""
        if self.finite_measurements:
            bandit_state = {
                meas: {
                    'V': b.V, 'V_tilde': b.V_tilde, 'b': b.b, 'theta': b.theta, 't': b.t,
                }
                for meas, b in self._bandits.items()
            }
        else:
            b = self.bandit
            bandit_state = {'V': b.V, 'V_tilde': b.V_tilde, 'b': b.b, 'theta': b.theta, 't': b.t}

        data = {
            'cost_history': self.cost_history,
            'bandit_state': bandit_state,
            'gamma': self._gamma, 'ridge_reg': self._ridge_reg,
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Checkpoint saved at step {len(self.cost_history)} -> {self.checkpoint_path}")

    def load_checkpoint(self):
        """Restore from a checkpoint. Returns True if successful."""
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            self.cost_history = data['cost_history']
            state = data['bandit_state']
            if self.finite_measurements:
                self._bandits = {}
                for meas, s in state.items():
                    agent = self._get_agent(meas)
                    agent.V, agent.V_tilde, agent.b, agent.theta, agent.t = (
                        s['V'], s['V_tilde'], s['b'], s['theta'], s['t'])
            else:
                self.bandit.V, self.bandit.V_tilde, self.bandit.b, self.bandit.theta, self.bandit.t = (
                    state['V'], state['V_tilde'], state['b'], state['theta'], state['t'])
            return True
        return False

    def reset(self):
        if self.finite_measurements:
            self._bandits = {}
        else:
            self.bandit = DLinUCB(d=self.d, ridge_reg=self._ridge_reg, gamma=self._gamma,
                                  delta=self._delta, sigma=self._sigma, S=self._S, L=self._L)
        self.cost_history = []

    def get_regret(self, display=False):
        """Stub so GamePlay's try/except around get_regret() doesn't need special-casing.
        (Not strictly necessary -- GamePlay already wraps this call in try/except -- but
        keeping it explicit avoids relying on that silently swallowing an AttributeError.)"""
        return (None, None, np.nan, None, None, None, None, None)
