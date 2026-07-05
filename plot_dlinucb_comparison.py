import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle, glob, re

plt.figure(figsize=(12, 7))

files = sorted(glob.glob('data/rps_101000_False/DLinUCB_lambda_*.pkl'))
for f in files:
    m = re.search(r'lambda_(.*)\.pkl', f)
    lam = m.group(1) if m else 'unknown'
    with open(f, 'rb') as fh:
        data = pickle.load(fh)
    plt.plot(data['iters'], data['average_costs'], label=f'λ = {lam}')

plt.xlabel('Iteration')
plt.ylabel('Rolling average cost (window = 1000)')
plt.title('D‑LinUCB regret – zoom around breakpoint')

# --- Zoom options (uncomment the ones you want) ---
plt.xlim(15000, 25000)        # just the breakpoint area
plt.ylim(-0.35, 0.1)          # typical cost range after learning

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()