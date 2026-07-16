# Boltzmann Learning — repository summary for Claude

## What this project is

`Boltzmann Learning` is a Python research project for sequential, cost-minimizing decisions under uncertain measurements. Its original algorithm is a Boltzmann/softmax learner (`LearningGame`) that receives full action-cost feedback. The current experimental work adds discounted linear UCB (D-LinUCB) plus a separate bandit-feedback comparison containing bandit Boltzmann and scikit-learn baselines. Most experiments use a non-stationary Rock–Paper–Scissors (RPS) opponent.

At every time step, an agent observes a measurement/context, selects an action, and receives a cost. The underlying state/opponent can depend on the past. `GamePlay` evaluates multiple agents separately against the same environment and persists their time-series outcomes as pickle files.

## Important caveats for working on this checkout

- This is a dirty working tree. Do not overwrite existing changes without inspecting them first. Modified tracked files are `examples/benchmark_methods.py`, `examples/rps_vs_bad_rng_nonstationary.py`, `examples/run_baselines.py`, `src/dlinucb.py`, and `src/dlinucb_decision_maker.py`; the current summary, canonical/fair runners, `BanditLearningGame`, plotting scripts, checkpoints, logs, and figures are untracked local work.
- Many root files are untracked experiment artifacts; they are part of the local research history, not necessarily committed project assets.
- `data/` is ignored by Git and now contains ~386 MB of binary pickle outputs. `venv/` is a local virtual environment (~358 MB) and is not source code.
- Imports are path-sensitive: project scripts commonly insert both the repository root and `src/` into `sys.path`. Core modules currently use absolute imports such as `from decision_maker import DecisionMaker`, not package-relative imports.

## Repository layout

```text
boltzmann-learning/
├── src/                         core learning algorithms and interface
├── examples/                    RPS/EMBER experiments, baselines, notebooks, utilities
│   └── simulation_utils/         generic game loop and dataset-game abstraction
├── test/                        unit/theory tests
├── data/                        saved experiment results (pickle; ignored; binary)
├── *.pkl                        root D-LinUCB checkpoints (binary)
├── *.out                        captured experiment stdout logs
├── plot_*.py                    one-off scripts to visualize saved results
├── fig_*.png, plot_check.png    generated figures
├── setup.py                     packaging/dependencies
├── README.md                    project overview and installation directions
├── LICENSE.txt, CONTACT-INFO.txt
├── .vscode/, .idea/             editor configuration/metadata
└── venv/                        local virtual environment; exclude from analysis
```

## Core implementation (`src/`)

- `src/__init__.py` — empty package initializer.
- `src/decision_maker.py` — abstract `DecisionMaker` contract. Any agent must implement `get_action(measurement, time, **kwargs)` and `update_energies(measurement, costs, time, **kwargs)`.
- `src/learning_games.py` — primary `LearningGame` Boltzmann learner. It maintains energy estimates per action/context, applies optional exponential time decay, computes a Boltzmann action distribution from inverse temperature, samples actions, updates from action costs, and calculates regret/entropy/bounds. It supports finite contexts and continuous contexts represented as distributions over the supplied measurement set. `get_random_integer` samples an index from a probability vector.
- `src/dlinucb.py` — standalone discounted linear UCB implementation based on Russac, Vernade, and Cappé (2019). `DLinUCB` maintains discounted design matrices `V`/`V_tilde`, reward vector `b`, parameter estimate `theta`, and time. It computes the confidence radius (`_beta`), selects the action feature with the largest UCB, and recursively updates with a reward. Its ridge parameter is named `ridge_reg` (not `lam`) to distinguish it from the experiment's swept forgetting-rate lambda.
- `src/dlinucb_decision_maker.py` — adapts `DLinUCB` to the project’s decision-maker/game-loop API. For finite contexts it lazily creates one `n_actions`-dimensional bandit per measurement rather than a single `(actions × measurements)`-dimensional bandit. This avoids an overly large global exploration bonus in sparse one-hot contexts. It maps costs to negative scaled rewards, records chosen-action costs, supports reset/load/save, and periodically checkpoints. Default checkpoints include the method name to prevent lambda-sweep agents from overwriting one another. Its `get_regret` is intentionally a compatibility stub. Its public ridge keyword is also `ridge_reg`.
- `src/bandit_learning_game.py` — `BanditLearningGame(LearningGame)`, used only for the fair bandit experiment. It updates the energy, bound bookkeeping, and cumulative-cost bookkeeping using only the chosen action’s realized cost; all unchosen action energies merely decay. It requires `action` in `update_energies`.
- `src/LearningGames.egg-info/` — generated editable-install metadata: `PKG-INFO`, dependency/top-level-package metadata, and source manifest; not authoritative source code.
- `src/__pycache__/` and `src/.DS_Store` — generated/interpreter/OS artifacts; ignore.

## Shared experiment utilities (`examples/simulation_utils/`)

- `__init__.py` — empty package initializer.
- `utils.py` — shared infrastructure:
  - `GamePlay` runs every decision maker for a specified horizon, fetches measurements, gets actions, executes the game, updates the agent, reports timing/progress, computes rolling averages, and serializes each method’s result dictionary.
  - Plot helpers for individual simulations/comparisons and binary policies.
  - `generate_probabilities_matrix`, which turns a probability matrix into measurement labels/probability representations.
- `dataset_game.py` — abstract base for games backed by fixed data/opponent sequences. It tracks time and measurements, defines an abstract cost, exposes the current measurement, and runs one-agent plays while returning selected cost, all action costs, and opponent action.
- `__pycache__/` — generated bytecode; ignore.

## Benchmark and experiment code (`examples/`)

- `benchmark_methods.py` — baseline agents:
  - `SklearnModel`: trains a supplied scikit-learn estimator on a sliding window of observations; supports one-hot contextual embeddings or raw features and chooses low-predicted-cost actions.
  - `MultiArmBandit`: basic epsilon-greedy, softmax, or Thompson-sampling multi-arm baseline.
  - `BayesianEstimator`: discrete-context/action Bayesian cost estimator.
- `bandit_sklearn_model.py` — `BanditSklearnModel(SklearnModel)`, a bandit-feedback baseline. It uses epsilon-greedy actions and stores only the cost of the chosen action. It retrains on a sliding window; MLP estimators are warm-started to reduce optimizer-noise oscillation, while SVC is normally refit.
- `run_canonical.py` — **primary full-information RPS experiment going forward.** It targets Anderson & Hespanha Section 6.1: five-symbol histories (243 contexts), two 150-action i.i.d. opponent sequences (seeds 11/7), a switch at 20,000, and 101,000 rounds. It runs the Bayesian estimator, three full-information Boltzmann configurations, full-information SVM/MLP windowed baselines, and the seven-value D-LinUCB forgetting sweep. It writes `data/canonical_101000_False/`. The SVM/MLP retrain cadence (1,000) and non-best Boltzmann configurations are documented, unverified paper assumptions.
- `run_canonical_seed42.py` — seed-variation of the canonical runner, using opponent seeds 42/43 and writing `data/canonical_seed42_101000_False/`.
- `run_bandit_fair.py` — **primary fair bandit-feedback RPS experiment.** It uses the same paper-style RPS process as the canonical runner, but runs three `BanditLearningGame` configurations, `BanditSVM`, `BanditMLP`, and the D-LinUCB sweep. The sklearn methods use a 5,000-step window (deliberately larger than the 1,000 used by per-context methods) because one global model otherwise has too little per-context/action bandit data. It writes `data/bandit_fair_101000/`.
- `run_baselines.py` — older combined runner, retained for local history and writing `data/rps_101000_False/`. It now contains bandit sklearn methods but still passes the obsolete `lam=` keyword to `DLinUCBDecisionMaker`; do not use it without changing that keyword to `ridge_reg=`. It also uses settings superseded by the canonical/fair runners.
- `reproduce_paper.py` — older best-effort full-information paper reproduction, superseded by `run_canonical.py`; it omits D-LinUCB and writes `data/paper_reproduction_101000_False/`.
- `run_bandit_baselines.py` and root `run_bandits_only.py` — older RPS runners for only the bandit sklearn baselines.
- `run_tuned_dlinucb.py` — legacy D-LinUCB tuning run. It likewise uses the obsolete `lam=` wrapper keyword and needs `ridge_reg=` before reuse.
- `rps_vs_bad_rng_nonstationary.py` — older/alternate non-stationary RPS script, currently focused on D-LinUCB. It now writes `data/SCRATCH_rps_...` to avoid colliding with real results, but also uses the obsolete `lam=` keyword and is not a current entry point.
- `rps_selfplay.py` — RPS game class that supports two learning players and runs a self-play demonstration.
- `rps_vs_bad_rng_nonstationary.py.bak` — backup of the non-stationary RPS script; historical reference only.
- `debugging_final_run.py` and `test_debug.py` — short/debug RPS experiment drivers combining Boltzmann, D-LinUCB, Bayesian, full-information sklearn, and bandit sklearn methods.
- `malware_classification.py` — EMBER malware-classification experiment. Defines `EMBERClassifierGame`, data loading/normalization, threshold optimization under weighted false-positive/false-negative cost, and a LightGBM-driven experiment. It requires externally downloaded EMBER data and can produce very large local datasets.
- `ember_plot.py` — loads/preprocesses saved EMBER outcomes and draws comparison plots.
- `extract_ember_meta.py` — small utility to inspect/extract EMBER metadata JSON.
- `rps_plot.py` — loads persisted RPS outcomes, reshapes them into tabular form, and plots method comparison data.
- `graphshortrun1.py` — simple historical plotting script for a pickled short-run output.
- `BS_selfplay.ipynb`, `rps_selfplay.ipynb`, `rps_vs_bad_rng.ipynb`, `rps_vs_fixed.ipynb`, `rps_vs_round_robin.ipynb` — interactive demonstrations of Boltzmann learning in RPS/self-play, random-opponent, fixed-opponent, and round-robin-opponent settings. Treat notebook outputs as supplemental; the `.py` sources are easier to review.
- `examples/data/` — duplicate/subset saved RPS pickle results used by examples; binary, not code.
- `examples/__pycache__/` and `.DS_Store` — generated artifacts; ignore.

## Tests (`test/`)

- `test_learning_games.py` — comprehensive `unittest` coverage for finite and continuous measurement learning games: action selection, energy updates, regret behavior, bounds/extrema, stationary-policy form, and theorem-related checks.
- `test_theory.py` — a reverse-Jensen theoretical check.
- `test_utils.py` — checks `generate_probabilities_matrix`.
- `test/__pycache__/` — generated bytecode; ignore.

## Root-level plotting scripts and figures

These scripts are one-off analysis tools. Most force the `TkAgg` Matplotlib backend, read `data/.../*.pkl`, and plot rolling average costs, often with a vertical marker at the opponent change around step 20,000.

- `plot_check.py` / `plot_check.png` — quick sanity comparison/check output.
- `plot_comparison.py` — generic experiment comparison.
- `plot_dlinucb.py` — plots one D-LinUCB result.
- `plot_dlinucb_all.py` — plots a D-LinUCB lambda sweep.
- `plot_dlinucb_comparison.py` — compares D-LinUCB result collections/configurations.
- `plot_final_clean.py` — produces the cleaned final figures: `fig_dlinucb.png`, `fig_boltzmann.png`, `fig_fullinfo.png`, and `fig_best.png`.
- `plot_final_comparison.py`, `plot_final_separate.py`, `plot_full_results_debug.py` — variants for grouped/full/debug result comparisons.
- `plot_single_lambda.py` — plots one selected lambda output.
- `plot_test.py` — plots every pickle result in `data/test_debug/`.
- `plot_final.py` — reusable result plotter for the older `data/rps_101000_False/` collection. It separates D-LinUCB, Boltzmann, bandit, and other methods into ordered method-family panels and writes rolling- and cumulative-average figures. It handles both pre- and post-Matplotlib-3.9 colormap APIs. `plot_final.py.bak` is a historical backup.
- `plot_canonical.py` and `plot_final_results_week4.py` — plotting helpers for canonical/older comparison sets; the latter displays grouped interactive rolling-cost panels.
- `plot_bandit_fair.py` and `plot_final_bandit_comparison.py` — fair-bandit result plotting, including rolling/cumulative figures such as `fig_bandit_fair_{rolling,cumulative}.png` and `fig_all_bandits.png`.
- `confirm.py` — tiny diagnostic that loads `DLinUCB_lambda_1e-02.pkl` and prints the post-switch cost autocorrelation at lag 150.
- `fig_all_bandits.png`, `fig_bandit_fair.png`, `fig_best.png`, `fig_boltzmann.png`, `fig_dlinucb.png`, `fig_dlinucb_sweep.png`, `fig_fullinfo.png`, `fig_paper_repro_cumulative.png`, `fig_paper_repro_rolling.png`, and `plot_check.png` — generated raster plots; inspect visually instead of treating them as source.

## Data, checkpoints, and logs

All `.pkl` files are Python pickle binaries. Do not unpickle untrusted files. Experiment-result pickles written by `GamePlay` typically include iteration values and rolling average costs (the plot scripts read `data['iters']` and `data['average_costs']`). D-LinUCB checkpoints instead store cost history, serialized bandit state, `gamma`, and `ridge_reg`.

- Root `DLinUCB_lambda_{0e+00,1e+01,1e+00,1e-01,1e-02,1e-03,1e-04}_checkpoint.pkl` and `dlinucb_checkpoint_DLinUCB_lambda_{...}.pkl` — saved D-LinUCB partial-state checkpoints. The latter uses the wrapper’s newer namespaced filename convention.
- `data/canonical_101000_False/` and `data/canonical_seed42_101000_False/` — current full-information canonical RPS result sets (default and seed-42/43 replication).
- `data/bandit_fair_101000/` — current fair bandit-feedback RPS result set. `data/bandit_fair_seed42_101000/`, `data/bandit_fair_101000_before_windowfix/`, and `data/bandit_fair_svmfix_101000/` are seed variation and historical diagnostic/fix result sets; do not pool them indiscriminately.
- `data/rps_101000_False/` — older 101k combined results: Boltzmann, D-LinUCB, and bandit sklearn methods. It is useful for historical plots but is superseded by the canonical/fair split.
- `data/rps_101000_False_backup/` — older/backup experiment set, including `BanditMLP`, `BanditSVM`, a wider D-LinUCB lambda sweep, four Boltzmann outputs, and `superseded/` outputs.
- `data/rps_101000_False_badwindow/` — a separate result collection with the same general methods but an unsuitable/experimental window configuration; preserve it as a non-comparable diagnostic set.
- `data/rps_2000_False/` — short 2,000-round smoke/debug set: Bayesian, four Boltzmann configurations, D-LinUCB, MLP, and SVC.
- `data/rps_bandit_101000_False/` — 101k RPS results for four Boltzmann configurations plus bandit MLP and SVC.
- `data/rps_final/` — consolidated final set: four Boltzmann outputs, D-LinUCB lambdas 0/1e+01/1e+00/1e-01/1e-02/1e-03/1e-04, and fixed bandit MLP/SVC outputs.
- `data/test_debug/` — debugging outputs: one Boltzmann, one D-LinUCB, and test bandit MLP/SVC result.
- `data/paper_reproduction_101000_False/` — results from `examples/reproduce_paper.py`: Bayesian, three paper-oriented Boltzmann configurations, and full-information MLP/SVM outputs.
- `examples/data/rps_101000_False/` and `examples/data/rps_bandit_101000_False/` — selected duplicate outputs used by example code.
- `bandits_only.out`, `bandits_v2.out`, `baselines.out`, `debugging_run.out`, `dlinucb_organized.out`, `full_fixed_run.out`, `full_run_final.out`, `full_run_final_v2.out`, `paper_repro.out`, and `tuned_dlinucb.out` — captured run logs with parameters, progress, checkpoints, and timing. `full_run_final_v2.out` reflects the newer namespaced checkpoints; `paper_repro.out` records the paper-reproduction run. `dlinucb.out` only records a failed `nohup: python: No such file or directory` launch.
- `debugging_final_run.py` at the repository root is an empty file.

## Project metadata and configuration

- `README.md` — high-level problem formulation, installation instructions (`pip install -e .`), RPS/EMBER example descriptions, and contacts. Some documented paths/names are stale (for example it references a `doc/` directory that is absent and older test naming).
- `setup.py` — package name `LearningGames`, version `0.1`, Python >=3.9. Runtime dependencies: NumPy, Matplotlib, SciPy, pandas, scikit-learn, LightGBM. Development extra: pytest.
- `LICENSE.txt` — UCSB academic/nonprofit-use license; commercial transfer/use requires permission.
- `CONTACT-INFO.txt` — UCSB/Joao Hespanha contact information and copyright note.
- `bandit_README.txt` — implementation note describing the five current D-LinUCB/bandit baseline files and the per-context D-LinUCB fix, checkpoint namespace fix, early-action baseline fix, and bandit-feedback/warm-start design.
- `.gitignore` — ignores data, caches, build output, docs builds, backups, profiling files, and OS artifacts.
- `.vscode/settings.json` — editor settings.
- `.idea/` — PyCharm project/module/inspection/VCS metadata; not runtime code.
- `.DS_Store` files — macOS Finder metadata; ignore.
- Root `__pycache__/` — generated bytecode for `plot_final.py`; ignore.

## Recommended orientation for a follow-up task

1. Start with `src/learning_games.py`, then `examples/simulation_utils/utils.py`, to understand the original agent/game API.
2. Read `src/dlinucb.py` and `src/dlinucb_decision_maker.py` for the current D-LinUCB implementation.
3. Read `examples/run_canonical.py` for the paper-oriented full-information benchmark, then `examples/run_bandit_fair.py` with `src/bandit_learning_game.py` and `examples/bandit_sklearn_model.py` for the fair comparison.
4. Use `bandit_README.txt` to understand the local D-LinUCB/bandit implementation decisions; its listed five-file bundle predates the canonical/fair-runner split.
5. Treat files in `data/`, figures, checkpoints, logs, caches, IDE folders, and `venv/` as derived artifacts unless a task explicitly involves reproducing/analyzing results.
