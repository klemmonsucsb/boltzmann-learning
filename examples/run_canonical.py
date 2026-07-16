"""
CANONICAL experiment script. This is the one script for the RPS comparison
going forward -- replaces run_baselines.py and reproduce_paper.py.

Everything here matches Anderson & Hespanha, "Learning with contextual
information in non-stationary environments," Section 6.1, EXCEPT for the one
addition: D-LinUCB (Russac, Vernade & Cappe, NeurIPS 2019), which is not part
of the original paper and is the actual contribution of this project. No
other deviations from the paper are intentional -- if you spot one, it's a
bug, not a design choice, and should be fixed or flagged, not left in.

Matches the paper on:
  - length_measurement = 5 (243 contexts)
  - opponent sequence: 150 i.i.d. random actions (rng.choice), not a
    forced-uniform permutation
  - switch_time = 20_000
  - Bayesian estimator baseline included
  - Boltzmann learning: paper's reported best config (lambda=1e-3, beta=1),
    plus a small-lambda and large-lambda config matching the paper's
    qualitative description of slow-recovery / high-regret behavior
  - SVM/MLP: FULL INFORMATION (see all 3 action costs per step), trained on
    a sliding window of the last 1,000 episodes -- matches the paper exactly.
    NOT bandit-restricted; that variant is a separate, later experiment and
    is intentionally excluded from this canonical script per project scope.

Deviates from the paper (the one necessary addition):
  - D-LinUCB, swept across lambda in [10, 1, 0.1, 0.01, 0.001, 0.0001, 0],
    using the per-context bandit design (see dlinucb_decision_maker.py).

UNVERIFIED ASSUMPTIONS (paper doesn't specify these numerically for RPS):
  - SVM/MLP retrain frequency: set to every 1,000 steps. The paper states
    the window is 1k episodes but does not state retrain frequency for RPS
    (only for the separate malware experiment, where it's 10k).
  - The exact (lambda, beta) values behind each curve in the paper's Figure
    1(a) aren't tabulated -- only the best config and qualitative behavior
    of the others are described in text.
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import os
import numpy as np
from collections import OrderedDict
from sklearn import svm, neural_network

from examples.benchmark_methods import BayesianEstimator, SklearnModel
from examples.simulation_utils.utils import GamePlay
from learning_games import LearningGame
from dlinucb_decision_maker import DLinUCBDecisionMaker


class RPSVsBadRNG:
    action_set = ["R", "P", "S"]

    def __init__(self, action_sequence, action_sequence_2, sequence_change_idx, length_measurement=1):
        self.action_sequence = action_sequence
        self.action_sequence_2 = action_sequence_2
        self.sequence_change_idx = sequence_change_idx
        self.time_counter = 0
        self.last_action = len(self.action_sequence) - 1
        measurement_set = self.action_set
        for i in range(length_measurement - 1):
            new_measurement_set = set()
            for m in measurement_set:
                for a in self.action_set:
                    new_measurement_set.add(m + a)
            measurement_set = new_measurement_set
        print("Measurement set with {:d} elements".format(len(measurement_set)))
        self.measurement_set = list(measurement_set)
        self.measurement = length_measurement * "S"

    def cost(self, p1_action, p2_action) -> float:
        if p1_action == p2_action:
            return 0
        if (p1_action == "R" and p2_action == "S") or (p1_action == "P" and p2_action == "R") or (
                p1_action == "S" and p2_action == "P"):
            return -1
        return +1

    def get_measurement(self):
        return self.measurement, None

    def play(self, p1_action):
        self.time_counter += 1
        self.last_action += 1
        if self.last_action >= len(self.action_sequence):
            self.last_action = 0
        if self.time_counter < self.sequence_change_idx:
            p2_action = self.action_sequence[self.last_action]
        else:
            p2_action = self.action_sequence_2[self.last_action]
        self.measurement = self.measurement[1:] + p2_action
        cost = self.cost(p1_action, p2_action)
        all_costs = OrderedDict([(a, self.cost(a, p2_action)) for a in self.action_set])
        return cost, all_costs, p2_action


if __name__ == '__main__':
    M: int = 101_000
    length_measurement: int = 5
    switch_time = 20_000
    label_to_action = {'R': 'P', 'P': 'S', 'S': 'R'}

    rng = np.random.default_rng(11)
    action_sequence = rng.choice(["R", "P", "S"], size=150)
    rng2 = np.random.default_rng(7)
    action_sequence2 = rng2.choice(["R", "P", "S"], size=150)

    measurement_to_label: bool = False
    data_window: int = 1_000
    update_freq: int = 1_000
    hidden_layer_sizes = (30, 30)
    max_train_iter = 500
    random_state = 1

    print("action sequence:", action_sequence)
    game = RPSVsBadRNG(action_sequence=action_sequence, length_measurement=length_measurement,
                        action_sequence_2=action_sequence2, sequence_change_idx=switch_time)

    methods = []

    # --- Bayesian estimator (paper's slow-recovery reference baseline) ---
    bayesian = BayesianEstimator(action_set=game.action_set, measurement_set=game.measurement_set)
    bayesian.name = 'Bayesian'
    methods.append(bayesian)

    # --- Boltzmann learning: paper's best config + slow/high-regret references ---
    lg_best = LearningGame(game.action_set, measurement_set=game.measurement_set,
                            decay_rate=1e-3, inverse_temperature=1.0, seed=0)
    lg_best.reset()
    lg_best.name = 'Boltzmann lambda=1e-03 beta=1e+00 (paper best)'
    methods.append(lg_best)

    lg_small = LearningGame(game.action_set, measurement_set=game.measurement_set,
                             decay_rate=1e-4, inverse_temperature=1.0, seed=0)
    lg_small.reset()
    lg_small.name = 'Boltzmann lambda=1e-04 beta=1e+00 (small lambda, slow recovery)'
    methods.append(lg_small)

    lg_large = LearningGame(game.action_set, measurement_set=game.measurement_set,
                             decay_rate=1e1, inverse_temperature=1e-2, seed=0)
    lg_large.reset()
    lg_large.name = 'Boltzmann lambda=1e+01 beta=1e-02 (large lambda, high regret)'
    methods.append(lg_large)

    # --- SVM / MLP: full-information, matching the paper exactly ---
    nn_model = neural_network.MLPRegressor(random_state=random_state, max_iter=max_train_iter,
                                            hidden_layer_sizes=hidden_layer_sizes)
    svm_model = svm.SVC(kernel='rbf')

    svm_dm = SklearnModel(window_size=data_window, action_set=game.action_set,
                          measurement_set=game.measurement_set, raw_measurement=False,
                          measurement_to_label=measurement_to_label, finite_measurement=True,
                          policy_map=label_to_action, update_frequency=update_freq, model=svm_model)
    svm_dm.name = 'SVM (full-info, paper-matched)'

    mlp_dm = SklearnModel(window_size=data_window, action_set=game.action_set,
                          measurement_set=game.measurement_set, raw_measurement=False,
                          measurement_to_label=measurement_to_label, finite_measurement=True,
                          policy_map=label_to_action, update_frequency=update_freq, model=nn_model)
    mlp_dm.name = 'MLP (full-info, paper-matched)'

    methods.append(svm_dm)
    methods.append(mlp_dm)

    # --- D-LinUCB: the one addition beyond the paper ---
    # delta=0.05 (not 0.5): Theorem 1's bound holds with probability 1-delta;
    # delta=0.5 would make that guarantee a coin flip, not meaningful.
    # S=sqrt(3) (not 1.0): S bounds ||theta*||_2. Each per-context theta has
    # 3 components (one reward estimate per action), each in [-1,1], so the
    # worst-case norm is sqrt(1^2+1^2+1^2)=sqrt(3), not 1.
    dlinucb_lambdas = [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 0.0]
    for lam in dlinucb_lambdas:
        gamma = np.exp(-lam) if lam > 0 else 1.0
        dlinucb = DLinUCBDecisionMaker(
            action_set=game.action_set, measurement_set=game.measurement_set,
            finite_measurements=True, gamma=gamma, ridge_reg=0.001, delta=0.05,
            sigma=1.0, S=np.sqrt(3), L=1.0, cost_scale=1.0,
            name=f'DLinUCB_lambda_{lam:.0e}'
        )
        methods.append(dlinucb)

    gp = GamePlay(decision_makers=methods, game=game, horizon=M,
                  disp_results_per_iter=int(M / 10), binary_cont_measurement=False,
                  store_energy_hist=False)

    save_dir = f'data/canonical_{M}_{str(measurement_to_label)}'
    os.makedirs(save_dir, exist_ok=True)
    gp.play_games(save_dir)
