"""
Bandit-restricted Boltzmann learning.

The original LearningGame.update_energies() (learning_games.py) updates the
energy of EVERY action every round it's called for the matching measurement:

    for a in self._action_set:
        self.energy[m][a] = decay * (self.energy[m][a] + costs[a] * weight)

This requires knowing costs[a] for every action a, not just the one actually
played -- full information, same as the original SVM/MLP baselines. To make
a fair comparison against D-LinUCB (which only ever observes the cost of the
action it took), this subclass restricts the energy update so only the
CHOSEN action's energy incorporates the observed cost; every other action's
energy still decays (time still passes) but gets no new information added.

The bound-tracking (min_cost/max_cost) and the internal average_cost used for
get_regret() in the original class also assume full information (they loop
over costs[a] for all a). This subclass restricts those too, using only the
realized cost of the action actually taken -- so this class never touches
information about actions it didn't play, anywhere.
"""
import numpy as np
from learning_games import LearningGame


class BanditLearningGame(LearningGame):
    """Bandit-restricted version of LearningGame (Boltzmann learning)."""

    def update_energies(self, measurement, costs: dict, time: float = 0.0, **kwargs):
        action = kwargs.get('action')
        if action is None:
            raise ValueError("BanditLearningGame.update_energies needs 'action' kwarg.")

        realized_cost = costs[action]

        # Bound tracking: only the realized cost is bandit feedback.
        if realized_cost < self.min_cost:
            self.min_cost = realized_cost
        if realized_cost > self.max_cost:
            self.max_cost = realized_cost

        decay = np.exp(-self.decay_rate * (time - self.time_update))

        # Internal regret bookkeeping: use the single realized cost, not a
        # probability-weighted average over all actions' costs (that average
        # requires full information the bandit setting doesn't have).
        self.total_cost = decay * self.total_cost + realized_cost

        for m in self._measurement_set:
            if self.finite_measurements:
                weight = 1.0 if m == measurement else 0.0
            else:
                weight = measurement[m]

            for a in self._action_set:
                # Only the action actually taken contributes new cost
                # information; everything else just decays (time passes,
                # no new information added since we never observed its cost).
                if a == action:
                    self.energy[m][a] = decay * (self.energy[m][a] + realized_cost * weight)
                else:
                    self.energy[m][a] = decay * self.energy[m][a]

        self.normalization_sum = decay * self.normalization_sum + 1
        self.time_update = time
