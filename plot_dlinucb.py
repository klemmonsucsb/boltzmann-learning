import matplotlib
matplotlib.use('TkAgg')   # often fixes blank windows on macOS
import matplotlib.pyplot as plt
import pickle

with open('data/rps_101000_False/DLinUCBDecisionMaker.pkl', 'rb') as f:
    data = pickle.load(f)

print("Keys:", data.keys())
print("Length:", len(data['average_costs']))

plt.plot(data['iters'], data['average_costs'])
plt.xlabel('Iteration')
plt.ylabel('Rolling avg cost (window=1000)')
plt.title('D‑LinUCB λ=0.1')
plt.grid(True)
plt.show()