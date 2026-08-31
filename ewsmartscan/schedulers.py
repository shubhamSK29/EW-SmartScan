"""Deterministic dwell-selection policies."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .beliefs import BetaBernoulliBelief
from .coverage import CoverageTracker
from .environment import Observation


@dataclass(frozen=True)
class SmartScoreWeights:
    w_eig: float = 1.0
    w_coverage: float = 0.5
    w_activity: float = 2.0


@dataclass(frozen=True)
class DwellScore:
    dwell: int
    information_gain: float
    coverage: float
    activity_probability: float
    score: float
    uncertainty: float


class SmartScanScheduler:
    """Select dwells using information, coverage, and learned activity."""

    name = "EW-SmartScan"

    def __init__(
        self,
        n_dwells: int,
        weights: SmartScoreWeights = SmartScoreWeights(),
        coverage_horizon: int | None = None,
    ):
        self.n_dwells = n_dwells
        self.belief = BetaBernoulliBelief(n_dwells)
        self.coverage = CoverageTracker(n_dwells, coverage_horizon)
        self.weights = weights

    def score_all(self, step: int) -> pd.DataFrame:
        information_gain = self.belief.normalised_information_gain()
        coverage = self.coverage.coverage_score(step)
        activity = self.belief.activity_probability
        score = (
            self.weights.w_eig * information_gain
            + self.weights.w_coverage * coverage
            + self.weights.w_activity * activity
        )
        return pd.DataFrame(
            {
                "dwell": np.arange(self.n_dwells),
                "label": [f"D{i:02d}" for i in range(self.n_dwells)],
                "activity_probability": activity,
                "uncertainty": self.belief.uncertainty,
                "information_gain": information_gain,
                "information_gain_raw": self.belief.expected_information_gain(),
                "coverage": coverage,
                "score": score,
            }
        )

    def select(self, step: int) -> int:
        scores = self.score_all(step)
        return int(np.argmax(scores["score"].to_numpy()))

    def update(self, observation: Observation) -> None:
        self.belief.update(observation.dwell, observation.hit)
        self.coverage.observe(observation.dwell, observation.step)

    def reset(self) -> None:
        """Forget everything learned so far."""
        self.belief = BetaBernoulliBelief(self.n_dwells)
        self.coverage = CoverageTracker(self.n_dwells, self.coverage.horizon_steps)


class RandomScheduler:
    """Seeded random baseline."""

    name = "Random"

    def __init__(self, n_dwells: int, seed: int = 0):
        self.n_dwells = n_dwells
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def select(self, step: int) -> int:
        return int(self.rng.integers(self.n_dwells))

    def update(self, observation: Observation) -> None:
        return None

    def reset(self) -> None:
        """Restart the seeded stream so runs stay reproducible."""
        self.rng = np.random.default_rng(self.seed)


class RoundRobinScheduler:
    """Deterministic sequential baseline."""

    name = "Round Robin"

    def __init__(self, n_dwells: int):
        self.n_dwells = n_dwells

    def select(self, step: int) -> int:
        return int(step % self.n_dwells)

    def update(self, observation: Observation) -> None:
        return None

    def reset(self) -> None:
        return None
