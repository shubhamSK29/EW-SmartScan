"""Beta-Bernoulli activity beliefs used by the Smart Scan scheduler.

For each dwell, alpha and beta are the hit and miss counts with a
Beta(1, 1) prior.  The predictive activity probability is
``p = alpha / (alpha + beta)`` and its variance is
``alpha*beta / ((alpha+beta)**2 * (alpha+beta+1))``.  Expected information
gain is ``H(p) - [p*H(p_hit) + (1-p)*H(p_miss)]``.
"""

import numpy as np


def binary_entropy(probability: np.ndarray | float) -> np.ndarray | float:
    """Return binary entropy in bits, defining 0*log2(0) as zero."""
    values = np.asarray(probability, dtype=float)
    clipped = np.clip(values, 0.0, 1.0)
    safe = np.clip(clipped, np.finfo(float).tiny, 1.0)
    entropy = -safe * np.log2(safe) - (1.0 - safe) * np.log2(
        np.clip(1.0 - safe, np.finfo(float).tiny, 1.0)
    )
    entropy = np.where((clipped == 0.0) | (clipped == 1.0), 0.0, entropy)
    return float(entropy) if entropy.ndim == 0 else entropy


class BetaBernoulliBelief:
    """Vectorised Beta-Bernoulli beliefs for a collection of dwells."""

    def __init__(self, n_dwells: int):
        self.alpha = np.ones(n_dwells, dtype=float)
        self.beta = np.ones(n_dwells, dtype=float)

    @property
    def activity_probability(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> np.ndarray:
        total = self.alpha + self.beta
        return self.alpha * self.beta / (total**2 * (total + 1.0))

    @property
    def uncertainty(self) -> np.ndarray:
        return np.sqrt(self.variance)

    @property
    def n_observations(self) -> np.ndarray:
        return self.alpha + self.beta - 2.0

    def update(self, i: int, hit: bool) -> None:
        """Update one dwell with its observed hit or miss."""
        if hit:
            self.alpha[i] += 1.0
        else:
            self.beta[i] += 1.0

    def expected_information_gain(self) -> np.ndarray:
        """Return expected reduction in predictive Bernoulli entropy."""
        total = self.alpha + self.beta
        p = self.activity_probability
        p_hit = (self.alpha + 1.0) / (total + 1.0)
        p_miss = self.alpha / (total + 1.0)
        expected = binary_entropy(p) - (
            p * binary_entropy(p_hit) + (1.0 - p) * binary_entropy(p_miss)
        )
        return np.maximum(np.asarray(expected, dtype=float), 0.0)

    def normalised_information_gain(self) -> np.ndarray:
        """Scale expected information gain relative to the prior value."""
        return np.clip(self.expected_information_gain() / EIG_PRIOR, 0.0, 1.0)


def _prior_eig() -> float:
    alpha = 1.0
    beta = 1.0
    total = alpha + beta
    p = alpha / total
    p_hit = (alpha + 1.0) / (total + 1.0)
    p_miss = alpha / (total + 1.0)
    return float(
        binary_entropy(p)
        - p * binary_entropy(p_hit)
        - (1.0 - p) * binary_entropy(p_miss)
    )


EIG_PRIOR = _prior_eig()
