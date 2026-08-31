"""Coverage and recency tracking for dwell scheduling."""

import numpy as np


class CoverageTracker:
    """Track how long it has been since each dwell was observed."""

    def __init__(self, n_dwells: int, horizon_steps: int | None = None):
        self.n_dwells = n_dwells
        self.horizon_steps = horizon_steps if horizon_steps is not None else n_dwells
        self.last_observed_step = np.full(n_dwells, -1, dtype=int)

    def observe(self, i: int, step: int) -> None:
        self.last_observed_step[i] = step

    def staleness(self, current_step: int) -> np.ndarray:
        unseen = self.last_observed_step < 0
        values = current_step - self.last_observed_step
        return np.where(unseen, current_step + 1, values)

    def coverage_score(self, current_step: int) -> np.ndarray:
        scores = np.clip(self.staleness(current_step) / self.horizon_steps, 0.0, 1.0)
        return np.where(self.last_observed_step < 0, 1.0, scores)
