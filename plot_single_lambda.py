import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

lam = '1e-03'   # change to any λ label like '1e-01', '0e+00', etc.
filepath = f'data/rps_101000_False/DLinUCB_lambda_{lam}.pkl'

with open(filepath, 'rb') as f:
    data = pickle.load(f)

plt.plot(data['iters'], data['average_costs'])
plt.xlabel('Iteration')
plt.ylabel('Rolling avg cost (window=1000)')
plt.title(f'D‑LinUCB λ = {lam}')
plt.grid(True)
plt.tight_layout()
plt.show()