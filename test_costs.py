import pickle

with open("../data/rps_40000_False/Boltzmann Learning | $\\lambda=1.0e-02, \\beta=1.0e+01$.pkl", "rb") as f:
    data = pickle.load(f)

costs = data['costs']
print("First 2000 avg cost:", costs[:2000].mean())
print("Last 2000 before switch avg cost:", costs[18000:20000].mean())
print("First 2000 after switch avg cost:", costs[20000:22000].mean())
print("Last 2000 avg cost:", costs[38000:].mean())