"""Frequency dwell definitions and nearest-centre frequency assignment."""

import numpy as np
import pandas as pd

from .data_loader import ReceiverConfig, TsrdDataset


class DwellPlan:
    """A regular or irregular list of frequency dwell bands."""

    def __init__(self, receiver: ReceiverConfig):
        self.n_dwells = len(receiver.dwell_centres_mhz)
        self.centres_mhz = np.asarray(receiver.dwell_centres_mhz, dtype=float)
        self.dwell_times_s = np.asarray(receiver.dwell_times_s, dtype=float)
        if self.n_dwells == 0:
            raise ValueError("receiver must define at least one dwell")
        if self.n_dwells > 1:
            half_spacing = float(np.median(np.diff(self.centres_mhz)) / 2.0)
        else:
            half_spacing = float(receiver.bandwidth_mhz / 2.0)
        self.half_spacing_mhz = half_spacing
        self.edges = np.concatenate(
            (
                [self.centres_mhz[0] - half_spacing],
                (self.centres_mhz[:-1] + self.centres_mhz[1:]) / 2.0,
                [self.centres_mhz[-1] + half_spacing],
            )
        )

    def label(self, i: int) -> str:
        self._check_index(i)
        return f"D{i:02d}"

    def bounds_mhz(self, i: int) -> tuple[float, float]:
        self._check_index(i)
        return float(self.edges[i]), float(self.edges[i + 1])

    def _check_index(self, i: int) -> None:
        if not 0 <= i < self.n_dwells:
            raise IndexError(f"dwell index out of range: {i}")

    def map_frequencies(self, freqs_mhz: np.ndarray) -> np.ndarray:
        """Assign frequencies to nearest centres, or -1 when out of range."""
        values = np.asarray(freqs_mhz, dtype=float)
        positions = np.searchsorted(self.centres_mhz, values, side="left")
        right = np.clip(positions, 0, self.n_dwells - 1)
        left = np.clip(positions - 1, 0, self.n_dwells - 1)
        choose_right = np.abs(values - self.centres_mhz[right]) < np.abs(
            values - self.centres_mhz[left]
        )
        assigned = np.where(choose_right, right, left).astype(int)
        outside = (values < self.edges[0]) | (values > self.edges[-1])
        assigned[outside] = -1
        return assigned

    def assign(self, dataset: TsrdDataset) -> pd.Series:
        """Return one dwell index for every PDW in ``dataset``."""
        return pd.Series(
            self.map_frequencies(dataset.pdws["frequency_mhz"].to_numpy()),
            index=dataset.pdws.index,
            name="dwell",
            dtype="int64",
        )
