import sys
sys.path.insert(0, 'src')
from learning_games_bandit import LearningGame

action_set = ['R', 'P', 'S']
lg = LearningGame(action_set, measurement_set=['test'], decay_rate=0.0, inverse_temperature=1.0, seed=0)

# Force a known starting state: equal energies, so equal probabilities (1/3 each)
print("Initial energies:", lg.energy['test'])

# Manually construct a bandit-feedback costs dict: only 'R' has a cost, rest are None
costs = {'R': -1.0, 'P': None, 'S': None}

lg.update_energies(measurement='test', costs=costs, time=1.0, action='R', bandit_correction="importance_weight")

print("Energies after one importance-weighted update:", lg.energy['test'])
print("Expected: R's energy should be -1.0 / (1/3) = -3.0, since all three started equiprobable")