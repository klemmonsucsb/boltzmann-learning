import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle, glob

# Load D‑LinUCB for λ=1e-03
with open('data/rps_101000_False/DLinUCB_lambda_1e-03.pkl','rb') as f:
    dlin = pickle.load(f)
plt.plot(dlin['iters'], dlin['average_costs'], label='D‑LinUCB λ=1e-03')

# Load MLP and SVM
for name in ['MLPRegressor','SVC']:
    with open(f'data/rps_101000_False/{name}.pkl','rb') as f:
        d = pickle.load(f)
    plt.plot(d['iters'], d['average_costs'], label=name)

# Find the Boltzmann file that used λ=1e-03 (if you generated it)
boltz_file = glob.glob('data/rps_101000_False/Boltzmann Learning | $*lambda=1.0e-03*.pkl')
if boltz_file:
    with open(boltz_file[0],'rb') as f:
        b = pickle.load(f)
    plt.plot(b['iters'], b['average_costs'], label='Boltzmann λ=1e-03, β=…')

plt.xlabel('Iteration'); plt.ylabel('Rolling avg cost')
plt.legend(); plt.grid(True); plt.tight_layout()
# plt.xlim(15000, 30000); plt.ylim(-1.0, -0.4)
plt.show()