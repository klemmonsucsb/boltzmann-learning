import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle, glob, os

# --- Helper: load and plot if file exists ---
def plot_if_exists(filename, label, **kwargs):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        plt.plot(data['iters'], data['average_costs'], label=label, **kwargs)
    else:
        print(f'File not found: {filename}')

# --- D-LinUCB (λ=0.001) – original regularisation ---
plot_if_exists('data/rps_101000_False/DLinUCB_lambda_1e-03.pkl',
               'D‑LinUCB λ=0.001, lam=1.0', linewidth=2)

# --- D-LinUCB (λ=0.001) – tuned regularisation ---
plot_if_exists('data/rps_101000_False/DLinUCB_lambda_1e-03_lam0.01.pkl',
               'D‑LinUCB λ=0.001, lam=0.01', linewidth=2, linestyle='--')

"""# --- MLP and SVM (full information) ---
for name in ['MLPRegressor', 'SVC']:
    plot_if_exists(f'data/rps_101000_False/{name}.pkl', name)
"""
# --- MLP and SVM (bandit feedback) ---
# Note: the files are named with '_bandit' suffix
plot_if_exists('data/rps_bandit_101000_False/MLPRegressor_bandit.pkl',
               'MLPRegressor (bandit)', linestyle='--')
plot_if_exists('data/rps_bandit_101000_False/SVC_bandit.pkl',
               'SVC (bandit)', linestyle='--')

# --- Bayesian ---
plot_if_exists('data/rps_101000_False/BayesianEstimator.pkl',
               'Bayesian (stationary)')

# --- All available Boltzmann learners (bandit) ---
boltz_files = sorted(glob.glob('data/rps_bandit_101000_False/Boltzmann Learning*'))
for f in boltz_files:
    with open(f, 'rb') as fh:
        b = pickle.load(fh)
    # extract λ and β from filename for the legend
    label = f.replace('Boltzmann Learning | ', '').replace('.pkl', '')
    plt.plot(b['iters'], b['average_costs'], label=f'Boltzmann {label}')

# --- Formatting ---
plt.xlabel('Iteration')
plt.ylabel('Rolling average cost (window=1000)')
plt.title('Algorithm comparison on RPS non‑stationary environment')
plt.legend()
plt.grid(True)

# Optional zoom around breakpoint
plt.xlim(15000, 30000)
plt.ylim(-1.0, 1.0)

plt.tight_layout()
plt.show()