import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle, os, glob, numpy as np

folder = 'data/rps_101000_False'
switch_time = 20_000

files = sorted(glob.glob(os.path.join(folder, '*.pkl')))

dlins, boltzs, bandits = [], [], []
for fpath in files:
    fname = os.path.basename(fpath).replace('.pkl', '')
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
    if 'DLinUCB' in fname and 'lam0.01' not in fname:
        dlins.append((fname, data))
    elif 'Boltzmann' in fname:
        boltzs.append((fname, data))
    elif 'Bandit' in fname:
        bandits.append((fname, data))

import itertools

# --- FIGURE 1: DLinUCB sweep ---
dl_colors = itertools.cycle(plt.cm.tab10.colors)
plt.figure(figsize=(12, 6))
for name, data in dlins:
    lam = name.split('_')[-1]
    plt.plot(data['iters'], data['average_costs'],
             color=next(dl_colors), linewidth=1.5, label=f'λ = {lam}')
plt.axvline(switch_time, color='red', linestyle='--', alpha=0.7, linewidth=1)
plt.xlabel('Iteration')
plt.ylabel('Rolling average cost (window=1000)')
plt.title('D‑LinUCB – Effect of Forgetting Rate λ')
plt.legend(fontsize=9, ncol=2)
plt.grid(True, alpha=0.3)
plt.ylim(-1.05, 0.55)
plt.tight_layout()
plt.savefig('fig_dlinucb_sweep.png', dpi=150, bbox_inches='tight')
plt.show()

# --- FIGURE 2: Best bandit comparison ---
plt.figure(figsize=(12, 6))
# Best DLinUCB
for name, data in dlins:
    if '1e-03' in name:
        plt.plot(data['iters'], data['average_costs'], color='#1f77b4',
                 linewidth=2.5, label='D‑LinUCB λ=0.001')
# Bandit MLP
for name, data in bandits:
    if 'MLP' in name:
        plt.plot(data['iters'], data['average_costs'], color='#2ca02c',
                 linewidth=2, linestyle='--', label='Bandit MLP (ε=0.05)')
# Bandit SVM
for name, data in bandits:
    if 'SVM' in name:
        plt.plot(data['iters'], data['average_costs'], color='#ff7f0e',
                 linewidth=2, linestyle='--', label='Bandit SVM (ε=0.05)')
# Best Boltzmann
for name, data in boltzs:
    if '1.0e-02' in name:
        plt.plot(data['iters'], data['average_costs'], color='#d62728',
                 linewidth=1.5, linestyle=':', label='Boltzmann (λ=0.01, β=10)')
plt.axvline(switch_time, color='red', linestyle='--', alpha=0.7, linewidth=1)
plt.xlabel('Iteration')
plt.ylabel('Rolling average cost (window=1000)')
plt.title('Bandit‑Feedback Comparison (Fair)')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim(-1.05, 0.55)
plt.tight_layout()
plt.savefig('fig_bandit_fair.png', dpi=150, bbox_inches='tight')
plt.show()

# --- FIGURE 3: Everything together ---
plt.figure(figsize=(16, 8))
dl_colors2 = itertools.cycle(plt.cm.tab10.colors)
for name, data in dlins:
    lam = name.split('_')[-1]
    plt.plot(data['iters'], data['average_costs'],
             color=next(dl_colors2), linewidth=1.2, label=f'DLinUCB λ={lam}')
bd_colors = ['#e377c2', '#17becf']
for i, (name, data) in enumerate(bandits):
    plt.plot(data['iters'], data['average_costs'],
             color=bd_colors[i], linewidth=2.5, linestyle='--', label=name)
b_colors = itertools.cycle(plt.cm.Set2.colors)
for name, data in boltzs:
    short = name.replace('Boltzmann Learning | ', '')[:45]
    plt.plot(data['iters'], data['average_costs'],
             color=next(b_colors), linewidth=1, linestyle=':', alpha=0.6, label=short)
plt.axvline(switch_time, color='red', linestyle='--', alpha=0.7, linewidth=1)
plt.xlabel('Iteration')
plt.ylabel('Rolling average cost (window=1000)')
plt.title('All Bandit Algorithms')
plt.legend(fontsize=6, bbox_to_anchor=(1.04, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.ylim(-1.05, 0.55)
plt.tight_layout()
plt.savefig('fig_all_bandits.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved: fig_dlinucb_sweep.png, fig_bandit_fair.png, fig_all_bandits.png")
