"""
Plotting script for run_bandit_fair.py's output.
Run from repo root: python plot_bandit_fair.py
"""
import os, glob, pickle
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "data/bandit_fair_101000"
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


def group_label(name):
    if name.startswith("DLinUCB"):
        return "D-LinUCB"
    if name.startswith("BanditBoltzmann"):
        return "Bandit Boltzmann"
    if name.startswith("BanditSVM") or name.startswith("BanditMLP"):
        return "Bandit SVM / MLP"
    return "Other"


def main():
    results = load_all(DATA_DIR)
    if not results:
        raise SystemExit(f"No .pkl files found in {DATA_DIR} -- did the run finish?")

    print(f"Loaded {len(results)} result files:")
    for name, costs in results.items():
        print(f"  {name}: {len(costs)} steps, mean cost = {costs.mean():.4f}")

    groups = {}
    for name in results:
        groups.setdefault(group_label(name), []).append(name)

    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    color_map = dict(zip(sorted(results.keys()), colors))

    fig, axes = plt.subplots(len(groups), 1, figsize=(11, 4 * len(groups)), sharex=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, (group, names) in zip(axes, groups.items()):
        for name in sorted(names):
            costs = results[name]
            rm = rolling_mean(costs, WINDOW)
            x = np.arange(len(rm)) + WINDOW // 2
            ax.plot(x, rm, label=name, color=color_map[name], linewidth=1.2)
        ax.axvline(SWITCH_TIME, color="red", linestyle="--", alpha=0.6, label="opponent switch")
        ax.set_title(group)
        ax.set_ylabel(f"rolling avg cost (w={WINDOW})")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig("fig_bandit_fair_rolling.png", dpi=150)
    print("Saved fig_bandit_fair_rolling.png")

    fig2, ax2 = plt.subplots(figsize=(11, 6))
    for name, costs in sorted(results.items()):
        cum_avg = np.cumsum(costs) / np.arange(1, len(costs) + 1)
        ax2.plot(cum_avg, label=name, color=color_map[name], linewidth=1.1)
    ax2.axvline(SWITCH_TIME, color="red", linestyle="--", alpha=0.6, label="opponent switch")
    ax2.set_xlabel("step")
    ax2.set_ylabel("cumulative average cost")
    ax2.set_title("Bandit-fair comparison: Boltzmann / SVM / MLP / D-LinUCB (all bandit feedback)")
    ax2.legend(fontsize=6, ncol=3)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("fig_bandit_fair_cumulative.png", dpi=150)
    print("Saved fig_bandit_fair_cumulative.png")


if __name__ == "__main__":
    main()
