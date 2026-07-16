import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle, os, glob, numpy as np

folder = 'data/rps_101000_False'
switch_time = 20_000

files = sorted(glob.glob(os.path.join(folder, '*.pkl')))

# Group files
dlins, boltzs, others = [], [], []
for fpath in files:
    fname = os.path.basename(fpath).replace('.pkl', '')
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
    if 'DLinUCB' in fname and 'lam0.01' not in fname:  # skip old tuned file
        dlins.append((fname, data))
    elif 'Boltzmann' in fname:
        boltzs.append((fname, data))
    elif 'MLPRegressor' in fname or 'SVC' in fname:
        others.append((fname, data))

# Colour maps
import itertools
dl_colors = itertools.cycle(plt.cm.tab10.colors)
b_colors = itertools.cycle(plt.cm.Set2.colors)
o_colors = itertools.cycle(plt.cm.Dark2.colors)

# --- FIGURE 1: DLinUCB only ---
plt.figure(figsize=(12, 6))
for name, data in dlins:
    # Extract lambda from filename
    lam = name.split('_')[-1]
    plt.plot(data['iters'], data['average_costs'],
             color=next(dl_colors), linewidth=1.5, label=f'λ = {lam}')
plt.axvline(switch_time, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Opponent change')
plt.xlabel('Iteration')
plt.ylabel('Rolling average cost (window=1000)')
plt.title('D‑LinUCB – Effect of Forgetting Rate λ')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.ylim(-1.05, 0.55)
plt.tight_layout()
plt.savefig('fig_dlinucb.png', dpi=150, bbox_inches='tight')
print("Saved fig_dlinucb.png")
plt.show()

# --- FIGURE 2: Boltzmann only ---
plt.figure(figsize=(10, 5))
for name, data in boltzs:
    # Short label
    label = name.replace('Boltzmann Learning | ', '')[:40]
    plt.plot(data['iters'], data['average_costs'],
             color=next(b_colors), linewidth=1.5, linestyle='--', label=label)
plt.axvline(switch_time, color='red', linestyle='--', alpha=0.7, linewidth=1)
plt.xlabel('Iteration')
plt.ylabel('Rolling average cost (window=1000)')
plt.title('Boltzmann Learning Baselines')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.ylim(-1.05, 0.55)
plt.tight_layout()
plt.savefig('fig_boltzmann.png', dpi=150, bbox_inches='tight')
print("Saved fig_boltzmann.png")
plt.show()

# --- FIGURE 3: Full‑information MLP & SVC ---
plt.figure(figsize=(10, 5))
for name, data in others:
    plt.plot(data['iters'], data['average_costs'],
             color=next(o_colors), linewidth=2, label=name)
plt.axvline(switch_time, color='red', linestyle='--', alpha=0.7, linewidth=1)
plt.xlabel('Iteration')
plt.ylabel('Rolling average cost (window=1000)')
plt.title('Full‑Information Baselines (unfair advantage)')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.ylim(-1.05, 0.55)
plt.tight_layout()
plt.savefig('fig_fullinfo.png', dpi=150, bbox_inches='tight')
print("Saved fig_fullinfo.png")
plt.show()

# --- FIGURE 4: Combined best ---
plt.figure(figsize=(14, 7))
# Best DLinUCB
for name, data in dlins:
    if '1e-03' in name and 'lam0.01' not in name:
        plt.plot(data['iters'], data['average_costs'], color='#1f77b4',
                 linewidth=2.5, linestyle='-', label='D‑LinUCB λ=0.001')
# Best Boltzmann
for name, data in boltzs:
    if '1.0e-02' in name:
        plt.plot(data['iters'], data['average_costs'], color='#ff7f0e',
                 linewidth=2, linestyle='--', label='Boltzmann (best)')
# MLP
for name, data in others:
    if 'MLP' in name:
        plt.plot(data['iters'], data['average_costs'], color='#2ca02c',
                 linewidth=2, linestyle=':', label='MLP (full‑info)')
plt.axvline(switch_time, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Change')
plt.xlabel('Iteration')
plt.ylabel('Rolling average cost (window=1000)')
plt.title('Best Performers Compared')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim(-1.05, 0.55)
plt.tight_layout()
plt.savefig('fig_best.png', dpi=150, bbox_inches='tight')
print("Saved fig_best.png")
plt.show()

print("\nAll figures saved: fig_dlinucb.png, fig_boltzmann.png, fig_fullinfo.png, fig_best.png")
