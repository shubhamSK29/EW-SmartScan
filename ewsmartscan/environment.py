"""Leakage-free replay of pulse observations from a loaded collection.

Two replay modes are available.

``band_local`` (default) gives every dwell its own observation timeline: each
band's recorded activity is partitioned into consecutive windows of that
band's dwell time, and a per-dwell pointer consumes one window per visit.  This
is the honest use of the evidence in the file, because the collection is not
spectrum ground truth -- it is the output of a scanning receiver that visited
each of the 36 bands on a fixed sweep of ``sum(dwell_times_s)`` (2.15 s).  A
band's pulses only exist in the file for the intervals when that receiver was
tuned to it, so a band is *unobserved* at every other instant and reporting a
miss there is a recording artifact rather than an absence of activity.  In this
mode the learnable per-band hit rate equals the band's intrinsic occupancy, and
the scheduling question ("which bands are worth receiver visits?") is well
posed.

One scheduling decision allocates a *dwell budget* rather than a single
receiver dwell: ``visit_duration_s`` (default 0.4 s) is the on-band time a
visit spends on the chosen band, i.e. ``round(visit_duration_s /
dwell_times_s[d])`` consecutive receiver dwell windows.  This is the decision
granularity of the scheduler, not a change to the receiver's dwell time -- the
windows are still the recorded dwell length, and every band gets the same
on-band time per decision, which keeps scheduler comparisons fair.  Because one
50-100 ms window only catches a scanning emitter 3-16 % of the time, a longer
budget is what makes the per-band difference learnable within a few hundred
decisions.  ``visit_duration_s=None`` consumes exactly one window per visit.

``wallclock`` replays a single global clock into the collection.  It is kept
only to demonstrate the artifact: a round-robin scheduler shares the 2.15 s
recording sweep period, so it phase-locks onto the recorded dwell windows and
scores far above any policy that spends its visits differently.
"""

from dataclasses import dataclass

import numpy as np

from .data_loader import TsrdDataset
from .dwells import DwellPlan

BAND_LOCAL = "band_local"
WALLCLOCK = "wallclock"
MODES = (BAND_LOCAL, WALLCLOCK)


@dataclass(frozen=True)
class Observation:
    """The information exposed to a scheduler after one dwell window."""

    step: int
    dwell: int
    sim_time_s: float
    window_start_s: float
    dwell_time_s: float
    visit_duration_s: float
    hit: bool
    n_pulses: int
    emitter_ids: tuple[int, ...]
    mean_amplitude_dbm: float | None
    mean_pulse_width_us: float | None
    mean_aoa_deg: float | None


class ReplayEnvironment:
    """Replay time windows from the dataset without exposing future PDWs."""

    def __init__(
        self,
        dataset: TsrdDataset,
        dwell_plan: DwellPlan,
        start_time_s: float = 0.0,
        sensitivity_dbm: float | None = None,
        mode: str = BAND_LOCAL,
        visit_duration_s: float | None = 0.4,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.dataset = dataset
        self.dwell_plan = dwell_plan
        self.start_time_s = float(start_time_s)
        self.sensitivity_dbm = sensitivity_dbm
        self.mode = mode
        self.visit_duration_s = visit_duration_s
        self.sim_time_s = self.start_time_s
        self.wraps = 0
        dwell_indices = dwell_plan.assign(dataset).to_numpy()
        frame = dataset.pdws
        self._times: list[np.ndarray] = []
        self._emitters: list[np.ndarray] = []
        self._amplitudes: list[np.ndarray] = []
        self._pulse_widths: list[np.ndarray] = []
        self._aoas: list[np.ndarray] = []
        for dwell in range(dwell_plan.n_dwells):
            mask = dwell_indices == dwell
            if sensitivity_dbm is not None:
                mask &= frame["amplitude_dbm"].to_numpy() >= sensitivity_dbm
            order = np.argsort(frame.loc[mask, "toa_s"].to_numpy(), kind="stable")
            selected = frame.loc[mask].iloc[order]
            self._times.append(selected["toa_s"].to_numpy(dtype=float))
            self._emitters.append(selected["emitter_id"].to_numpy(dtype=int))
            self._amplitudes.append(selected["amplitude_dbm"].to_numpy(dtype=float))
            self._pulse_widths.append(selected["pulse_width_us"].to_numpy(dtype=float))
            self._aoas.append(selected["aoa_deg"].to_numpy(dtype=float))
        self._window_bounds = [self._build_windows(dwell) for dwell in range(dwell_plan.n_dwells)]
        self.n_windows = np.array([len(lo) for lo, _ in self._window_bounds], dtype=int)
        self.windows_per_visit = np.array(
            [self._windows_per_visit(dwell) for dwell in range(dwell_plan.n_dwells)], dtype=int
        )
        self._pointer = self._initial_pointers()

    def _build_windows(self, dwell: int) -> tuple[np.ndarray, np.ndarray]:
        """Return per-window pulse index bounds for one dwell's own timeline."""
        dwell_time = float(self.dwell_plan.dwell_times_s[dwell])
        n_windows = max(int(np.floor(self.dataset.receiver.collection_time_s / dwell_time)), 1)
        edges = np.arange(n_windows + 1, dtype=float) * dwell_time
        bounds = np.searchsorted(self._times[dwell], edges, side="left")
        return bounds[:-1], bounds[1:]

    def _windows_per_visit(self, dwell: int) -> int:
        """Return how many consecutive windows one visit to ``dwell`` consumes."""
        if self.visit_duration_s is None:
            return 1
        if self.visit_duration_s <= 0:
            raise ValueError("visit_duration_s must be positive or None")
        dwell_time = float(self.dwell_plan.dwell_times_s[dwell])
        windows = int(round(self.visit_duration_s / dwell_time))
        return min(max(windows, 1), int(self.n_windows[dwell]))

    def _initial_pointers(self) -> np.ndarray:
        starts = self.start_time_s / self.dwell_plan.dwell_times_s
        return np.floor(starts).astype(int) % self.n_windows

    def reset(self) -> None:
        """Return replay time and every dwell pointer to the start position."""
        self.sim_time_s = self.start_time_s
        self.wraps = 0
        self._pointer = self._initial_pointers()

    def _wallclock_indices(self, dwell: int, start: float, end: float) -> np.ndarray:
        times = self._times[dwell]
        if end <= self.dataset.receiver.collection_time_s:
            lo = np.searchsorted(times, start, side="left")
            hi = np.searchsorted(times, end, side="left")
            return np.arange(lo, hi)
        first_lo = np.searchsorted(times, start, side="left")
        second_hi = np.searchsorted(
            times, end - self.dataset.receiver.collection_time_s, side="left"
        )
        return np.concatenate((np.arange(first_lo, len(times)), np.arange(0, second_hi)))

    def _band_local_indices(self, dwell: int, dwell_time: float) -> tuple[np.ndarray, float]:
        first = int(self._pointer[dwell])
        total = int(self.n_windows[dwell])
        count = int(self.windows_per_visit[dwell])
        lo, hi = self._window_bounds[dwell]
        last = first + count
        if last <= total:
            indices = np.arange(lo[first], hi[last - 1])
        else:
            wrapped = last - total
            indices = np.concatenate(
                (np.arange(lo[first], hi[total - 1]), np.arange(lo[0], hi[wrapped - 1]))
            )
            self.wraps += 1
        self._pointer[dwell] = last % total
        if last == total:
            self.wraps += 1
        return indices, first * dwell_time

    def observe(self, dwell: int, step: int) -> Observation:
        """Observe only the selected dwell during the current dwell window."""
        dwell_time = float(self.dwell_plan.dwell_times_s[dwell])
        start = self.sim_time_s
        end = start + dwell_time
        if self.mode == BAND_LOCAL:
            indices, window_start = self._band_local_indices(dwell, dwell_time)
            visit_duration = dwell_time * int(self.windows_per_visit[dwell])
            self.sim_time_s = start + visit_duration
        else:
            visit_duration = dwell_time
            indices = self._wallclock_indices(dwell, start, end)
            window_start = start
            self.sim_time_s = end
            if self.sim_time_s > self.dataset.receiver.collection_time_s:
                self.sim_time_s = self.start_time_s
                self.wraps += 1
        amplitudes = self._amplitudes[dwell][indices]
        pulse_widths = self._pulse_widths[dwell][indices]
        aoas = self._aoas[dwell][indices]
        emitters = self._emitters[dwell][indices]
        return Observation(
            step=step,
            dwell=dwell,
            sim_time_s=start,
            window_start_s=window_start,
            dwell_time_s=dwell_time,
            visit_duration_s=visit_duration,
            hit=bool(len(indices)),
            n_pulses=int(len(indices)),
            emitter_ids=tuple(int(item) for item in np.unique(emitters)),
            mean_amplitude_dbm=float(np.mean(amplitudes)) if len(indices) else None,
            mean_pulse_width_us=float(np.mean(pulse_widths)) if len(indices) else None,
            mean_aoa_deg=float(np.mean(aoas)) if len(indices) else None,
        )

    def n_pulses_per_dwell(self) -> np.ndarray:
        """Return the number of recorded pulses in each dwell band."""
        return np.array([len(times) for times in self._times], dtype=int)

    def ground_truth_hit_rate(self, dwell_time_s: float) -> np.ndarray:
        """Compute fixed-window hit rates independently of scheduler replay."""
        if dwell_time_s <= 0:
            raise ValueError("dwell_time_s must be positive")
        n_windows = int(np.floor(self.dataset.receiver.collection_time_s / dwell_time_s))
        if n_windows == 0:
            return np.zeros(self.dwell_plan.n_dwells)
        edges = np.arange(n_windows + 1, dtype=float) * dwell_time_s
        rates = np.zeros(self.dwell_plan.n_dwells)
        for dwell in range(self.dwell_plan.n_dwells):
            bounds = np.searchsorted(self._times[dwell], edges, side="left")
            rates[dwell] = np.mean(bounds[1:] > bounds[:-1])
        return rates

    def ground_truth_occupancy(self) -> np.ndarray:
        """Return the hit rate a scheduler experiences with the configured visit."""
        rates = np.zeros(self.dwell_plan.n_dwells)
        for dwell, (lo, hi) in enumerate(self._window_bounds):
            count = int(self.windows_per_visit[dwell])
            n_visits = len(lo) // count
            starts = lo[: n_visits * count : count]
            ends = hi[count - 1 : n_visits * count : count]
            rates[dwell] = np.mean(ends > starts)
        return rates
