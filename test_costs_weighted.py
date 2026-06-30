import pickle

with open("../data/rps_40000_importanceweight/Boltzmann Learning | $\\lambda=1.0e-02, \\beta=1.0e+01$.pkl", "rb") as f:
    data_iw = pickle.load(f)

costs = data_iw['costs']
print("=== Importance-Weighted, λ=0.01 ===")
print("First 2000 avg cost:", costs[:2000].mean())
print("Last 2000 before switch avg cost:", costs[18000:20000].mean())
print("First 2000 after switch avg cost:", costs[20000:22000].mean())
print("Last 2000 avg cost:", costs[38000:].mean())

with open("../data/rps_40000_importanceweight/Boltzmann Learning | $\\lambda=1.0e-01, \\beta=1.0e+00$.pkl", "rb") as f:
    data_iw_2 = pickle.load(f)

costs2 = data_iw_2['costs']
print("\n=== Importance-Weighted, λ=0.1 ===")
print("First 2000 avg cost:", costs2[:2000].mean())
print("Last 2000 before switch avg cost:", costs2[18000:20000].mean())
print("First 2000 after switch avg cost:", costs2[20000:22000].mean())
print("Last 2000 avg cost:", costs2[38000:].mean())