import pickle
import numpy as np

with open("../data/rps_40000_False/Boltzmann Learning | $\\lambda=1.0e-01, \\beta=1.0e+00$.pkl", "rb") as f:
    data = pickle.load(f)

costs = data['costs']
print("First 2000 avg cost:", np.mean(costs[:2000]))
print("Last 2000 before switch avg cost:", np.mean(costs[18000:20000]))
print("First 2000 after switch avg cost:", np.mean(costs[20000:22000]))
print("Last 2000 avg cost:", np.mean(costs[38000:]))