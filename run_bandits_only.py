import sys, os
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "examples"))

import numpy as np
from collections import OrderedDict
from sklearn import svm, neural_network
from examples.simulation_utils.utils import GamePlay
from bandit_sklearn_model import BanditSklearnModel

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
        if p1_action == p2_action: return 0
        if (p1_action == "R" and p2_action == "S") or (p1_action == "P" and p2_action == "R") or (p1_action == "S" and p2_action == "P"):
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
    M = 101_000
    length_measurement = 5
    switch_time = 20_000
    data_window = 300          # effective window = 900
    update_freq = 1000         # retrain every 1000 steps (like original)
    rng = np.random.default_rng(11)
    action_sequence = rng.permutation(["R", "P", "S"] * 50)
    rng2 = np.random.default_rng(7)
    action_sequence2 = rng2.permutation(["R", "P", "S"] * 50)

    measurement_to_label = False
    label_to_action = {'R': 'P', 'P': 'S', 'S': 'R'}
    hidden_layer_sizes = (30, 30)
    max_train_iter = 500

    game = RPSVsBadRNG(action_sequence=action_sequence,
                       action_sequence_2=action_sequence2,
                       sequence_change_idx=switch_time,
                       length_measurement=length_measurement)

    methods = []

    # Bandit SVM
    svm_model = svm.SVC(kernel='rbf', random_state=None)
    bandit_svm = BanditSklearnModel(
        window_size=data_window,
        action_set=game.action_set,
        measurement_set=game.measurement_set,
        raw_measurement=False,
        measurement_to_label=measurement_to_label,
        finite_measurement=True,
        policy_map=label_to_action,
        update_frequency=update_freq,
        model=svm_model,
        eps=0.05
    )
    bandit_svm.name = 'BanditSVM'
    methods.append(bandit_svm)

    # Bandit MLP
    nn_model = neural_network.MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_train_iter,
        random_state=None
    )
    bandit_mlp = BanditSklearnModel(
        window_size=data_window,
        action_set=game.action_set,
        measurement_set=game.measurement_set,
        raw_measurement=False,
        measurement_to_label=measurement_to_label,
        finite_measurement=True,
        policy_map=label_to_action,
        update_frequency=update_freq,
        model=nn_model,
        eps=0.05
    )
    bandit_mlp.name = 'BanditMLP'
    methods.append(bandit_mlp)

    gp = GamePlay(decision_makers=methods, game=game, horizon=M,
                  disp_results_per_iter=int(M/10), binary_cont_measurement=False,
                  store_energy_hist=False)
    save_dir = 'data/rps_101000_False'
    os.makedirs(save_dir, exist_ok=True)
    gp.play_games(save_dir)
