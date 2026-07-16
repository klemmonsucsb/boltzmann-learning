"""
Bandit-FAIR comparison: every algorithm here only ever observes the cost of
the action it actually took -- Boltzmann, SVM, and MLP are all bandit-
restricted versions, put on equal footing with D-LinUCB (which is inherently
bandit-feedback by construction). This is the apples-to-apples comparison;
run_canonical.py (full-information, matching the paper exactly) is the
separate reference showing how much each method benefits from full info.

Same paper-matched setup otherwise: length_measurement=5 (243 contexts),
150 i.i.d. random opponent actions, switch_time=20_000, M=101_000.
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

from examples.bandit_sklearn_model import BanditSklearnModel
from examples.simulation_utils.utils import GamePlay
from bandit_learning_game import BanditLearningGame
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

    data_window: int = 1_000
    update_freq: int = 1_000
    hidden_layer_sizes = (30, 30)
    max_train_iter = 500
    random_state = 1

    print("action sequence:", action_sequence)
    game = RPSVsBadRNG(action_sequence=action_sequence, length_measurement=length_measurement,
                        action_sequence_2=action_sequence2, sequence_change_idx=switch_time)

    methods = []

    # --- Bandit-restricted Boltzmann: same 3 configs as run_canonical.py ---
    lg_best = BanditLearningGame(game.action_set, measurement_set=game.measurement_set,
                                  decay_rate=1e-3, inverse_temperature=1.0, seed=0)
    lg_best.reset()
    lg_best.name = 'BanditBoltzmann lambda=1e-03 beta=1e+00'
    methods.append(lg_best)

    lg_small = BanditLearningGame(game.action_set, measurement_set=game.measurement_set,
                                   decay_rate=1e-4, inverse_temperature=1.0, seed=0)
    lg_small.reset()
    lg_small.name = 'BanditBoltzmann lambda=1e-04 beta=1e+00'
    methods.append(lg_small)

    lg_large = BanditLearningGame(game.action_set, measurement_set=game.measurement_set,
                                   decay_rate=1e1, inverse_temperature=1e-2, seed=0)
    lg_large.reset()
    lg_large.name = 'BanditBoltzmann lambda=1e+01 beta=1e-02'
    methods.append(lg_large)

    # --- Bandit-restricted SVM / MLP ---
    # NOTE: bandit_data_window (5000) is intentionally larger than data_window
    # (1000, used elsewhere for DLinUCB-comparable timing). BanditSklearnModel
    # trains ONE global model across all 243 contexts x 3 actions from a single
    # window, unlike DLinUCB/Boltzmann which maintain 243 separate per-context
    # models. Bandit feedback yields only 1 sample/step (vs. 3/step for full
    # info), so at window=1000 there's under 1.4 samples per (context,action)
    # cell on average -- far too sparse for a stable RBF-SVM decision boundary,
    # which is what caused BanditSVM's persistent non-convergence. Verified via
    # smoke test (M=15000): mean cost and variance both improve monotonically
    # as window grows from 1000 to 8000; window=5000 is a practical middle
    # ground (real improvement without the heaviest retrain cost). Tradeoff:
    # slower adaptation to the step-20,000 switch, in exchange for stability.
    bandit_data_window = 5000

    svm_model = svm.SVC(kernel='rbf')
    bandit_svm = BanditSklearnModel(
        window_size=bandit_data_window, action_set=game.action_set, measurement_set=game.measurement_set,
        raw_measurement=False, measurement_to_label=False, finite_measurement=True,
        policy_map=label_to_action, update_frequency=update_freq, model=svm_model, eps=0.05)
    bandit_svm.name = 'BanditSVM'
    methods.append(bandit_svm)

    nn_model = neural_network.MLPRegressor(random_state=random_state, max_iter=max_train_iter,
                                            hidden_layer_sizes=hidden_layer_sizes)
    bandit_mlp = BanditSklearnModel(
        window_size=bandit_data_window, action_set=game.action_set, measurement_set=game.measurement_set,
        raw_measurement=False, measurement_to_label=False, finite_measurement=True,
        policy_map=label_to_action, update_frequency=update_freq, model=nn_model, eps=0.05)
    bandit_mlp.name = 'BanditMLP'
    methods.append(bandit_mlp)

    # --- D-LinUCB: bandit feedback by construction, same sweep as run_canonical.py ---
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

    save_dir = f'data/bandit_fair_{M}'
    os.makedirs(save_dir, exist_ok=True)
    gp.play_games(save_dir)
