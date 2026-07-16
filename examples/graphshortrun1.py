import pickle

with open('data/rps_2000_False/DLinUCBDecisionMaker.pkl', 'rb') as f:
    data = pickle.load(f)

print(data.keys())  # dict_keys(['iters', 'costs', 'average_costs', 'entropy', 'probs', 'energies'])


import matplotlib.pyplot as plt

plt.plot(data['iters'], data['average_costs'], label='D-LinUCB (M=2000)')
plt.xlabel('Iteration')
plt.ylabel('Average cost (rolling)')
plt.legend()
plt.grid()
plt.show()