"""
Reproduction of the RPS experiment from Anderson & Hespanha,
"Learning with contextual information in non-stationary environments," Sec 6.1.

Differences from run_baselines.py (intentional -- this script targets paper
fidelity, not the later bandit-restricted comparison):
  - Boltzmann learners see FULL information (all 3 action costs per step),
    matching the paper -- not bandit-restricted.
  - SVM/MLP are the original SklearnModel (full information), not
    BanditSklearnModel -- matching the paper's actual setup.
  - Includes the Bayesian estimator baseline, which the paper explicitly
    compares against.
  - length_measurement=5 (243 contexts, matches paper exactly).
  - Opponent sequence is 150 i.i.d. random draws (rng.choice), not a
    forced-uniform permutation -- matches "150 randomly selected actions."

KNOWN UNCERTAINTIES -- the paper does not fully specify these for the RPS
experiment, so treat this as best-effort, not a guaranteed exact match:
  - Exact retrain frequency for SVM/MLP in RPS (only stated for the malware
    example: every 10k steps). update_freq=1000 here is a reasonable guess,
    not confirmed against the paper text.
  - The exact (lambda, beta) pairs used to produce Figure 1(a)'s specific
    curves aren't listed as a table -- only the best (1e-3, 1) and the
    qualitative behavior of a "small lambda" and "large lambda" curve are
    described. The three configs below are chosen to match those qualitative
    descriptions, not verified against original hyperparameter values.
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
    length_measurement: int = 5          # FIXED: paper uses 5, not 3
    switch_time = 20_000
    label_to_action = {'R': 'P', 'P': 'S', 'S': 'R'}

    # FIXED: 150 i.i.d. random draws ("150 randomly selected actions"),
    # not a forced-uniform permutation of exactly 50 R/50 P/50 S.
    rng = np.random.default_rng(11)
    action_sequence = rng.choice(["R", "P", "S"], size=150)
    rng2 = np.random.default_rng(7)
    action_sequence2 = rng2.choice(["R", "P", "S"], size=150)

    measurement_to_label: bool = False   # regressor mode -- paper's best-performing approach
    data_window: int = 1_000             # paper: "last 1k episodes"
    update_freq: int = 1_000             # NOT explicitly specified by paper for RPS -- best guess
    hidden_layer_sizes = (30, 30)
    max_train_iter = 500
    svm_kernel = 'rbf'
    random_state = 1

    print("action sequence:", action_sequence)
    game = RPSVsBadRNG(action_sequence=action_sequence, length_measurement=length_measurement,
                        action_sequence_2=action_sequence2, sequence_change_idx=switch_time)

    methods = []

    # --- Bayesian estimator (paper's slow-recovery reference baseline) ---
    bayesian = BayesianEstimator(action_set=game.action_set, measurement_set=game.measurement_set)
    bayesian.name = 'Bayesian'
    methods.append(bayesian)

    # --- Boltzmann learning: three configs matching the paper's qualitative description ---
    # (1) the paper's reported best config
    lg_best = LearningGame(game.action_set, measurement_set=game.measurement_set,
                            decay_rate=1e-3, inverse_temperature=1.0, seed=0)
    lg_best.reset()
    lg_best.name = 'Boltzmann lambda=1e-03 beta=1e+00 (paper best)'
    methods.append(lg_best)

    # (2) small lambda -> paper: "slow recovery... closely follows the Bayesian estimator"
    lg_small = LearningGame(game.action_set, measurement_set=game.measurement_set,
                             decay_rate=1e-4, inverse_temperature=1.0, seed=0)
    lg_small.reset()
    lg_small.name = 'Boltzmann lambda=1e-04 beta=1e+00 (small lambda, slow recovery)'
    methods.append(lg_small)

    # (3) large lambda -> paper: "unable to produce sufficiently low regret... above -0.02"
    lg_large = LearningGame(game.action_set, measurement_set=game.measurement_set,
                             decay_rate=1e1, inverse_temperature=1e-2, seed=0)
    lg_large.reset()
    lg_large.name = 'Boltzmann lambda=1e+01 beta=1e-02 (large lambda, high regret)'
    methods.append(lg_large)

    # --- SVM / MLP: full-information, matching the paper (not bandit-restricted) ---
    if measurement_to_label:
        nn_model = neural_network.MLPClassifier(random_state=random_state, max_iter=max_train_iter,
                                                 hidden_layer_sizes=hidden_layer_sizes)
        svm_model = svm.SVC(kernel=svm_kernel)
    else:
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

    gp = GamePlay(decision_makers=methods, game=game, horizon=M,
                  disp_results_per_iter=int(M / 10), binary_cont_measurement=False,
                  store_energy_hist=False)

    save_dir = f'data/paper_reproduction_{M}_{str(measurement_to_label)}'
    os.makedirs(save_dir, exist_ok=True)
    gp.play_games(save_dir)