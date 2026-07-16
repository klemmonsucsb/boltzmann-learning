import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle, os, glob

folder = 'data/rps_101000_False'
switch_time = 20_000

# Load and group files
files = sorted(glob.glob(os.path.join(folder, '*.pkl')))
if not files:
    print(f"No files in {folder}")
    exit()

dlins, bandits, others = [], [], []
for fpath in files:
    fname = os.path.basename(fpath).replace('.pkl', '')
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
    if 'DLinUCB' in fname:
        dlins.append((fname, data))
    elif 'Bandit' in fname:
        bandits.append((fname, data))
    else:
        others.append((fname, data))

# Colour cycles
dl_colors = plt.cm.tab10.colors + plt.cm.tab20b.colors
bd_colors = plt.cm.Set2.colors
ot_colors = plt.cm.Pastel1.colors

import itertools
dl_cycle = itertools.cycle(dl_colors)
bd_cycle = itertools.cycle(bd_colors)
ot_cycle = itertools.cycle(ot_colors)

# Helper
def plot_group(items, cycle, title, ls='-', lw=1.5):
    plt.figure(figsize=(12, 6))
    for name, data in items:
        plt.plot(data['iters'], data['average_costs'],
                 color=next(cycle), linestyle=ls, linewidth=lw, label=name)
    plt.axvline(switch_time, color='red', linestyle='--', alpha=0.7, label='Change')
    plt.xlabel('Iteration'); plt.ylabel('Rolling avg cost')
    plt.title(title); plt.legend(fontsize=8); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

# Combined
plt.figure(figsize=(14, 7))
for name, data in dlins:
    plt.plot(data['iters'], data['average_costs'],
             color=next(dl_cycle), linestyle='-', linewidth=1.5, label=f'DLinUCB: {name.split("_")[-1]}')
for name, data in bandits:
    plt.plot(data['iters'], data['average_costs'],
             color=next(bd_cycle), linestyle=':', linewidth=2.0, label=name)
for name, data in others:
    plt.plot(data['iters'], data['average_costs'],
             color=next(ot_cycle), linestyle='--', linewidth=1.2, label=name)
plt.axvline(switch_time, color='red', linestyle='--', alpha=0.7)
plt.xlabel('Iteration'); plt.ylabel('Rolling avg cost')
plt.title('All algorithms')
plt.legend(fontsize=7, bbox_to_anchor=(1.04, 1), loc='upper left')
plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

# Separate figures
plot_group(dlins, dl_cycle, 'D‑LinUCB (λ sweep)', ls='-')
plot_group(bandits, bd_cycle, 'Bandit baselines', ls=':', lw=2.0)
plot_group(others, ot_cycle, 'Bayesian & others', ls='--')