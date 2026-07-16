import sys, os
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import numpy as np
from collections import OrderedDict
from dlinucb_decision_maker import DLinUCBDecisionMaker
from examples.simulation_utils.utils import GamePlay

# Reuse the RPSVsBadRNG class from the original script
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
    rng = np.random.default_rng(11)
    action_sequence = rng.permutation(["R", "P", "S"] * 50)
    rng2 = np.random.default_rng(7)
    action_sequence2 = rng2.permutation(["R", "P", "S"] * 50)

    game = RPSVsBadRNG(action_sequence=action_sequence, length_measurement=length_measurement,
                       action_sequence_2=action_sequence2, sequence_change_idx=switch_time)

    # Tuned D-LinUCB: same forgetting factor (γ = exp(-1e-3)), much lower ridge regularisation

    """
    Discounted LinUCB (Russac et al., 2019).

    Note on naming: this implementation's `lambda_forget` (forgetting rate)
    corresponds to the paper's discount factor gamma (gamma = exp(-lambda_forget)
    in some formulations, or set directly). This implementation's `lam`
    (ridge parameter) corresponds to the paper's lambda (regularization).
    Paper's tuning rule: gamma = 1 - (B_T / (d*T))**(2/3), which depends on
    the variation budget B_T and dimension d, not just T.
    """
    gamma = np.exp(-1e-3)
    dlinucb = DLinUCBDecisionMaker(
        action_set=game.action_set,
        measurement_set=game.measurement_set,
        finite_measurements=True,
        gamma=gamma,
        lam=0.01,                 # <-- lower regularisation
        delta=0.05,
        sigma=1.0,
        S=1.0,
        L=1.0,
        cost_scale=1.0,
        name='DLinUCB_lambda_1e-03_lam0.01'   # unique name
    )

    methods = [dlinucb]
    gp = GamePlay(decision_makers=methods, game=game, horizon=M,
                  disp_results_per_iter=int(M/10), binary_cont_measurement=False,
                  store_energy_hist=False)
    save_dir = f'data/rps_{M}_False'
    os.makedirs(save_dir, exist_ok=True)
    gp.play_games(save_dir)