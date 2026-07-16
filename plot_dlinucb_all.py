import matplotlib
matplotlib.use('TkAgg')          # works reliably on macOS
import matplotlib.pyplot as plt
import pickle, glob, re

plt.figure(figsize=(12,7))

for f in sorted(glob.glob('data/rps_101000_False/DLinUCB_lambda_*.pkl')):
    lam = re.search(r'lambda_(.*)\.pkl', f).group(1)
    with open(f, 'rb') as fh:
        data = pickle.load(fh)
    plt.plot(data['iters'], data['average_costs'], label=f'λ = {lam}')

plt.xlabel('Iteration')
plt.ylabel('Rolling average cost (window=1000)')
plt.title('D‑LinUCB – effect of forgetting factor')
plt.legend()
plt.grid(True)

# Uncomment to zoom around the breakpoint (step 20,000)
# plt.xlim(15000, 30000)
# plt.ylim(-1.0, -0.4)

plt.tight_layout()
plt.show()