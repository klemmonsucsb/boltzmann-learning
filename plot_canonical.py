"""
Plot the canonical experiment's results: paper baselines + DLinUCB.
Run from repo root: python plot_canonical.py
"""
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "data/canonical_101000_False"
WINDOW = 1000
SWITCH_TIME = 20_000

def load_all(data_dir):
    results = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.pkl"))):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, "rb") as f:
            d = pickle.load(f)
        results[name] = np.asarray(d["costs"])
    return results

def rolling_mean(x, w):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")

def group_of(name):
    # simple substring matching -- no regex, nothing that can silently
    # return None and crash a sort
    if name.startswith("DLinUCB"):
        return "DLinUCB (added beyond the paper)"
    if name.startswith("Boltzmann"):
        return "Boltzmann (paper)"
    return "Paper baselines (Bayesian / SVM / MLP)"

def main():
    results = load_all(DATA_DIR)
    if not results:
        raise SystemExit(f"No .pkl files in {DATA_DIR} -- did the run finish?")

    print(f"Loaded {len(results)} result files:")
    for name, costs in results.items():
        print(f"  {name}: {len(costs)} steps, mean cost = {costs.mean():.4f}")

    groups = {}
    for name in results:
        groups.setdefault(group_of(name), []).append(name)
    for g in groups:
        groups[g].sort()  # plain alphabetical -- stable, never crashes

    n_groups = len(groups)
    colors_cycle = plt.cm.tab20(np.linspace(0, 1, 20))

    # --- rolling average, one subplot per group ---
    fig, axes = plt.subplots(n_groups, 1, figsize=(11, 4 * n_groups), sharex=True)
    if n_groups == 1:
        axes = [axes]
    for ax, (group, names) in zip(axes, groups.items()):
        for i, name in enumerate(names):
            costs = results[name]
            rm = rolling_mean(costs, WINDOW)
            x = np.arange(len(rm)) + WINDOW // 2
            ax.plot(x, rm, label=name, color=colors_cycle[i % 20], linewidth=1.2)
        ax.axvline(SWITCH_TIME, color="red", linestyle="--", alpha=0.6, label="opponent switch")
        ax.set_title(group)
        ax.set_ylabel(f"rolling avg cost (w={WINDOW})")
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig("fig_canonical_rolling.png", dpi=150)
    print("Saved fig_canonical_rolling.png")

    # --- cumulative average, all on one plot ---
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    all_colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    for i, (name, costs) in enumerate(sorted(results.items())):
        cum_avg = np.cumsum(costs) / np.arange(1, len(costs) + 1)
        ax2.plot(cum_avg, label=name, color=all_colors[i], linewidth=1.1)
    ax2.axvline(SWITCH_TIME, color="red", linestyle="--", alpha=0.6, label="opponent switch")
    ax2.set_xlabel("step")
    ax2.set_ylabel("cumulative average cost")
    ax2.set_title("Cumulative average cost: paper baselines vs D-LinUCB")
    ax2.legend(fontsize=6.5, ncol=2, loc="upper right")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("fig_canonical_cumulative.png", dpi=150)
    print("Saved fig_canonical_cumulative.png")

if __name__ == "__main__":
    main()
