# Working D-LinUCB + bandit-baseline implementation

These are the 5 files that make up the current, tested implementation.
Copy them to the matching paths in your repo:

- `dlinucb.py` -> `src/dlinucb.py`
- `dlinucb_decision_maker.py` -> `src/dlinucb_decision_maker.py`
- `benchmark_methods.py` -> `examples/benchmark_methods.py`
- `bandit_sklearn_model.py` -> `examples/bandit_sklearn_model.py`
- `run_baselines.py` -> `examples/run_baselines.py`

## What's correct about each one

**dlinucb.py** — core algorithm. Never had a bug. Implements Algorithm 1 from
Russac, Vernade & Cappe (NeurIPS 2019) directly: weighted recursive least
squares with discount factor gamma = e^-lambda, confidence radius beta_t from
their Eq. 5, UCB action selection a^T theta_hat + beta_t * sqrt(a^T V^-1 Vtilde
V^-1 a).

**dlinucb_decision_maker.py** — wrapper connecting DLinUCB to the game loop.
FIXED: now keeps one small DLinUCB(d=3) per measurement/context (243 of them),
instead of one DLinUCB(d=729) over the whole one-hot space. The paper's beta_t
scales with ambient dimension d; a single 729-dim bandit pays a
729-dimensional exploration bonus on every action even though each context
only ever lives in a 3-dim slice, so it barely converges across 101k rounds.
Verified: with this fix, lambda=0.01 and lambda=0 (slow forgetting) clearly
beat lambda=1 (fast forgetting) even in a short smoke test — matches theory.
Also fixed: checkpoint files are now namespaced by `name`, so sweeping
multiple lambda values no longer overwrites the same checkpoint file.

**benchmark_methods.py** — original full-info SVM/MLP baseline class
(`SklearnModel`), used as the parent class for the bandit version below.
FIXED one bug: `get_action` used to `return 0, None, None` (the literal
integer 0) when the model hadn't been trained yet, instead of a real action
from `action_set` — this could silently corrupt cost comparisons early in a
run. Now returns `self.action_set[0]`.

**bandit_sklearn_model.py** — `BanditSklearnModel(SklearnModel)`, the fair
bandit-restricted baseline. Two things make this the correct version (after
an earlier partial_fit-based attempt was reconsidered and rejected — see
conversation for why):
  - Only the cost of the action actually taken is stored (`costs[action]`,
    not the whole `costs` dict) — bandit feedback, not full information.
  - Forgetting is via the sliding window (same as the original), not via
    resetting the model. For MLP (`warm_start=True`), each retrain calls
    `.fit()` on the current window starting from the previous solution
    instead of a fresh random init — this avoids the random-init-driven
    oscillation of the original design while still forgetting correctly
    (old data outside the window is never in the loss, warm-start or not).
    Verified empirically: on 5 sequential windows of *unchanged* synthetic
    data, full-reset-refit moved predictions on a fixed probe set ~5x more
    than warm-started refit (0.57 vs 0.11 avg abs change) -- most of the
    original oscillation was optimizer noise, not real adaptation.
  - SVC (kernel SVM) has no warm_start concept at all (it's a QP solve, not
    gradient descent) and is deterministic for fixed data, so it's left on
    plain refit-from-scratch each window -- that was never the source of
    oscillation.

**run_baselines.py** — main experiment script, `M=101_000`,
`length_measurement=5` (243 contexts), `switch_time=20_000`. Runs the 7-value
DLinUCB lambda sweep, the original 4 Boltzmann (lambda,beta) pairs, and
BanditSVM/BanditMLP. Saves to `data/rps_101000_False/`.

## Known separate issue -- not part of this bundle

`examples/rps_vs_bad_rng_nonstationary.py` writes to the SAME output path
(`data/rps_{M}_{measurement_to_label}`) as `run_baselines.py`, and has been
hand-edited to use `length_measurement=3` and a single lambda value. If it's
ever run again it will silently overwrite files in `data/rps_101000_False/`
with results from a different, non-comparable experiment. Either don't run
it, or repoint its save_dir before you do.