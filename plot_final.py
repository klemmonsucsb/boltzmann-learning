"""
Plot the rps_101000_False sweep results, split into panels by method family
so colors never repeat and the D-LinUCB / Boltzmann sweeps are ordered by
their hyperparameter (so you can see the sweep shape, not just spaghetti).

Usage:
    cd ~/boltzmann-learning
    python plot_final.py
Produces:
    fig_rolling_average.png       (one panel per method family)
    fig_cumulative_average.png    (same layout, cumulative average cost)
"""
import glob
import os
import pickle
import re

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

DATA_DIR = "data/rps_101000_False"
SWITCH_TIME = 20_000  # matches switch_time in run_baselines.py


def load_results(data_dir):
    results = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.pkl"))):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, "rb") as f:
            results[name] = pickle.load(f)
    return results


def parse_dlinucb_lambda(name):
    m = re.search(r"lambda_([\d.e+-]+)", name)
    return float(m.group(1)) if m else None


def parse_boltzmann_decay(name):
    m = re.search(r"\\lambda=([\d.e+-]+)", name)
    return float(m.group(1)) if m else None


def group_methods(results):
    groups = {"DLinUCB": [], "Boltzmann": [], "Bandit": [], "Other": []}
    for name in results:
        if name.startswith("DLinUCB"):
            groups["DLinUCB"].append(name)
        elif name.startswith("Boltzmann"):
            groups["Boltzmann"].append(name)
        elif name.startswith("Bandit"):
            groups["Bandit"].append(name)
        else:
            groups["Other"].append(name)
    groups["DLinUCB"].sort(key=parse_dlinucb_lambda)
    groups["Boltzmann"].sort(key=lambda n: parse_boltzmann_decay(n) if parse_boltzmann_decay(n) is not None else float("inf"))
    groups["Bandit"].sort()
    return {k: v for k, v in groups.items() if v}


def short_label(name):
    if name.startswith("Boltzmann"):
        # "Boltzmann Learning | $\lambda=1.0e-02, \beta=1.0e+01$" -> "Boltzmann lam=1e-02 beta=1e+01"
        m = re.search(r"\\lambda=([\d.e+-]+),\s*\\beta=([\d.e+-]+)", name)
        if m:
            return f"Boltzmann \u03bb={m.group(1)} \u03b2={m.group(2)}"
    return name


def get_cmap(name):
    """Works across matplotlib versions: cm.get_cmap() was removed in mpl>=3.9
    in favor of matplotlib.colormaps[name]."""
    try:
        return matplotlib.colormaps[name]
    except AttributeError:
        return cm.get_cmap(name)


def color_for_group(group_name, n):
    """Distinct, ordered colors per group so nothing repeats across panels."""
    if group_name == "DLinUCB":
        cmap = get_cmap("viridis")
    elif group_name == "Boltzmann":
        cmap = get_cmap("plasma")
    elif group_name == "Bandit":
        cmap = get_cmap("Dark2")
    else:
        cmap = get_cmap("tab10")
    if n == 1:
        return [cmap(0.5)]
    return [cmap(i / (n - 1)) for i in range(n)]


def plot_panel(ax, results, names, series_fn, group_name, ylabel, ylim=None):
    colors = color_for_group(group_name, len(names))
    for name, color in zip(names, colors):
        data = results[name]
        iters = np.array(data["iters"])
        y = series_fn(data)
        ax.plot(iters, y, label=short_label(name), color=color, linewidth=1.2)
    ax.axvline(SWITCH_TIME, color="black", linestyle="--", alpha=0.4, linewidth=1)
    ax.set_xlabel("t")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.set_title(group_name)
    ax.legend(fontsize=7, loc="best")


def rolling_series(data):
    return np.array(data["average_costs"])


def cumulative_series(data):
    costs = np.array(data["costs"])
    return np.cumsum(costs) / np.arange(1, len(costs) + 1)


def make_figure(results, groups, series_fn, ylabel, filename, ylim=None):
    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(6.5 * n_groups, 6), sharey=True)
    if n_groups == 1:
        axes = [axes]
    for ax, (group_name, names) in zip(axes, groups.items()):
        plot_panel(ax, results, names, series_fn, group_name, ylabel, ylim=ylim)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Saved {filename}")


def main():
    results = load_results(DATA_DIR)
    if not results:
        raise SystemExit(f"No .pkl files found in {DATA_DIR}")
    print(f"Loaded {len(results)} result files:")
    for name in results:
        print(" -", name)

    groups = group_methods(results)

    make_figure(results, groups, rolling_series,
                "rolling average cost (window=1000)",
                "fig_rolling_average.png", ylim=(-1.1, 1.1))

    make_figure(results, groups, cumulative_series,
                "cumulative average cost",
                "fig_cumulative_average.png")

    plt.show()


if __name__ == "__main__":
    main()