import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle, os, glob

folder = 'data/rps_final'
switch_time = 20_000

# ---------- Load all files and group them ----------
dlins = []
boltzs = []
bandits = []

for fpath in sorted(glob.glob(os.path.join(folder, '*.pkl'))):
    fname = os.path.basename(fpath).replace('.pkl', '')
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
    if 'DLinUCB' in fname:
        dlins.append((fname, data))
    elif 'Boltzmann' in fname:
        boltzs.append((fname, data))
    else:
        bandits.append((fname, data))

# ---------- Bright, distinct colour maps ----------
# Each group gets its own cycle.  We use tab10 + tab20b so colours
# are clearly different **within** each group AND across groups.
import itertools
dl_colors  = list(plt.cm.tab10.colors) + list(plt.cm.tab20b.colors)
b_colors   = list(plt.cm.Set1.colors) + list(plt.cm.Pastel1.colors)
bd_colors  = list(plt.cm.Dark2.colors) + list(plt.cm.Set2.colors)

# Cycle through the colours safely
dl_color_cycle  = itertools.cycle(dl_colors)
b_color_cycle   = itertools.cycle(b_colors)
bd_color_cycle  = itertools.cycle(bd_colors)

# ---------- Helper to plot one group ----------
def plot_group(items, color_cycle, title, linestyle='-', lw=1.8):
    for name, data in items:
        color = next(color_cycle)
        plt.plot(data['iters'], data['average_costs'],
                 color=color, linestyle=linestyle, linewidth=lw, label=name)
    plt.axvline(switch_time, color='red', linestyle='--', alpha=0.7, linewidth=1)
    plt.xlabel('Iteration')
    plt.ylabel('Rolling average cost (window=1000)')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

# ---------- Figure 1: Combined overview ----------
plt.figure(figsize=(14, 7))

# DLinUCB – solid, thick
for name, data in dlins:
    plt.plot(data['iters'], data['average_costs'],
             color=next(dl_color_cycle), linestyle='-', linewidth=1.5, label=name)
# Boltzmann – dashed
for name, data in boltzs:
    plt.plot(data['iters'], data['average_costs'],
             color=next(b_color_cycle), linestyle='--', linewidth=1.2, label=name)
# Bandit baselines – dotted, thick
for name, data in bandits:
    plt.plot(data['iters'], data['average_costs'],
             color=next(bd_color_cycle), linestyle=':', linewidth=2.0, label=name)

plt.axvline(switch_time, color='red', linestyle='--', alpha=0.7, linewidth=1)
plt.xlabel('Iteration')
plt.ylabel('Rolling average cost')
plt.title('All algorithms (debugged)')
plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=7)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ---------- Figure 2: D-LinUCB only ----------
plt.figure(figsize=(10, 5))
plot_group(dlins, dl_color_cycle, 'D‑LinUCB (varying λ)', linestyle='-')
plt.show()

# ---------- Figure 3: Boltzmann only ----------
plt.figure(figsize=(10, 5))
plot_group(boltzs, b_color_cycle, 'Boltzmann learners', linestyle='--')
plt.show()

# ---------- Figure 4: Bandit baselines only ----------
plt.figure(figsize=(10, 5))
plot_group(bandits, bd_color_cycle, 'Bandit‑restricted MLP & SVM', linestyle=':', lw=2.2)
plt.show()