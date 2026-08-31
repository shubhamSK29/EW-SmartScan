"""Tests for the small EW-SmartScan core demonstrator."""

import numpy as np
import pytest

from ewsmartscan.beliefs import BetaBernoulliBelief, binary_entropy
from ewsmartscan.coverage import CoverageTracker
from ewsmartscan.data_loader import DEFAULT_DATA_PATH, load_tsrd
from ewsmartscan.dwells import DwellPlan
from ewsmartscan.environment import ReplayEnvironment
from ewsmartscan.simulation import Simulation
from ewsmartscan.schedulers import (
    RandomScheduler,
    RoundRobinScheduler,
    SmartScanScheduler,
)


@pytest.fixture(scope="module")
def dataset():
    return load_tsrd(DEFAULT_DATA_PATH)


@pytest.fixture(scope="module")
def plan(dataset):
    return DwellPlan(dataset.receiver)


def test_loader(dataset):
    assert list(dataset.pdws.columns) == [
        "toa_us",
        "toa_s",
        "frequency_mhz",
        "pulse_width_us",
        "aoa_deg",
        "amplitude_dbm",
        "emitter_id",
    ]
    assert np.allclose(dataset.pdws.toa_s, dataset.pdws.toa_us / 1e6)
    assert len(dataset.pdws) == dataset.metadata["num_pulses"]
    assert len(dataset.receiver.dwell_centres_mhz) == 36
    assert dataset.emitters


def test_dwell_mapping(dataset, plan):
    assigned = plan.assign(dataset)
    assert (assigned >= 0).all()
    for dwell in range(plan.n_dwells):
        values = dataset.pdws.loc[assigned == dwell, "frequency_mhz"]
        if len(values):
            lo, hi = plan.bounds_mhz(dwell)
            assert values.min() >= lo
            assert values.max() <= hi
    counts = assigned.value_counts()
    for dwell in [1, 4, 13, 14, 15, *range(22, 36)]:
        assert counts.get(dwell, 0) == 0
    assert counts.get(18, 0) > 0


def test_belief_math():
    belief = BetaBernoulliBelief(1)
    assert belief.activity_probability[0] == 0.5
    belief.update(0, True)
    belief.update(0, True)
    belief.update(0, True)
    belief.update(0, False)
    assert belief.alpha[0] == 4
    assert belief.beta[0] == 2
    assert belief.activity_probability[0] == 4 / 6
    expected_variance = 4 * 2 / ((6**2) * 7)
    assert belief.variance[0] == pytest.approx(expected_variance)
    for alpha in [1, 2, 5, 20]:
        for beta in [1, 3, 20]:
            assert np.all(BetaBernoulliBelief(1).expected_information_gain() >= 0)
            test_belief = BetaBernoulliBelief(1)
            test_belief.alpha[0] = alpha
            test_belief.beta[0] = beta
            assert test_belief.expected_information_gain()[0] >= 0
    eigs = []
    for count in [1, 5, 20]:
        test_belief = BetaBernoulliBelief(1)
        test_belief.alpha[:] = count
        test_belief.beta[:] = count
        eigs.append(test_belief.expected_information_gain()[0])
    assert eigs[0] > eigs[1] > eigs[2]
    assert BetaBernoulliBelief(1).normalised_information_gain()[0] == pytest.approx(1.0)


def test_entropy():
    assert binary_entropy(0) == 0
    assert binary_entropy(1) == 0
    assert binary_entropy(0.5) == pytest.approx(1)


def test_coverage():
    coverage = CoverageTracker(3)
    assert coverage.coverage_score(0)[0] == 1.0
    coverage.observe(0, 0)
    assert coverage.coverage_score(0)[0] == 0.0
    assert coverage.staleness(3)[0] > coverage.staleness(1)[0]


@pytest.mark.parametrize("mode", ["band_local", "wallclock"])
def test_environment(dataset, plan, mode):
    """Single-window visits, so both modes advance the clock by one dwell time."""
    environment = ReplayEnvironment(dataset, plan, mode=mode, visit_duration_s=None)
    first = environment.observe(0, 0)
    assert first.sim_time_s == 0.0
    assert environment.sim_time_s == pytest.approx(plan.dwell_times_s[0])
    empty = ReplayEnvironment(dataset, plan, mode=mode, visit_duration_s=None).observe(1, 0)
    assert not empty.hit
    occupied = ReplayEnvironment(dataset, plan, mode=mode, visit_duration_s=None)
    observations = [occupied.observe(18, step) for step in range(20)]
    assert any(item.hit for item in observations)
    left = ReplayEnvironment(dataset, plan, mode=mode, visit_duration_s=None)
    right = ReplayEnvironment(dataset, plan, mode=mode, visit_duration_s=None)
    assert [left.observe(18, i) for i in range(20)] == [
        right.observe(18, i) for i in range(20)
    ]


def test_environment_rejects_unknown_mode(dataset, plan):
    with pytest.raises(ValueError):
        ReplayEnvironment(dataset, plan, mode="sideways")


def test_wallclock_reproduces_measured_occupancy(dataset, plan):
    environment = ReplayEnvironment(dataset, plan, mode="wallclock")
    rates = environment.ground_truth_hit_rate(0.05)
    assert rates[18] == pytest.approx(0.137, abs=0.002)
    assert rates[17] == pytest.approx(0.117, abs=0.002)
    assert rates[6] == pytest.approx(0.113, abs=0.002)
    assert rates[1] == 0.0


@pytest.mark.parametrize("visit_duration_s", [None, 0.4])
def test_band_local_experienced_rate_matches_occupancy(dataset, plan, visit_duration_s):
    environment = ReplayEnvironment(
        dataset, plan, mode="band_local", visit_duration_s=visit_duration_s
    )
    occupancy = environment.ground_truth_occupancy()
    for dwell in (17, 18, 6):
        environment.reset()
        visits = 200 if visit_duration_s is None else 60
        hits = sum(environment.observe(dwell, step).hit for step in range(visits))
        assert hits / visits == pytest.approx(occupancy[dwell], abs=0.07)


def test_visit_duration_consumes_a_dwell_budget(dataset, plan):
    environment = ReplayEnvironment(dataset, plan, mode="band_local", visit_duration_s=0.4)
    for dwell in (2, 18):
        dwell_time = float(plan.dwell_times_s[dwell])
        expected_windows = round(0.4 / dwell_time)
        assert int(environment.windows_per_visit[dwell]) == expected_windows
        environment.reset()
        before = environment.sim_time_s
        observation = environment.observe(dwell, 0)
        assert observation.visit_duration_s == pytest.approx(0.4)
        assert environment.sim_time_s - before == pytest.approx(0.4)
        assert int(environment._pointer[dwell]) == expected_windows


def test_single_window_visits_when_duration_is_none(dataset, plan):
    budgeted = ReplayEnvironment(dataset, plan, mode="band_local", visit_duration_s=0.4)
    single = ReplayEnvironment(dataset, plan, mode="band_local", visit_duration_s=None)
    assert single.windows_per_visit.tolist() == [1] * plan.n_dwells
    first = single.observe(18, 0)
    assert first.visit_duration_s == pytest.approx(plan.dwell_times_s[18])
    assert single.sim_time_s == pytest.approx(plan.dwell_times_s[18])
    assert budgeted.observe(18, 0).n_pulses >= first.n_pulses


def test_empty_dwell_never_hits_with_budgeted_visits(dataset, plan):
    environment = ReplayEnvironment(dataset, plan, mode="band_local", visit_duration_s=0.4)
    assert not any(environment.observe(1, step).hit for step in range(100))


def test_band_local_pointers_are_independent(dataset, plan):
    environment = ReplayEnvironment(dataset, plan, mode="band_local")
    reference = ReplayEnvironment(dataset, plan, mode="band_local")
    for step in range(5):
        environment.observe(17, step)
    moved = environment.observe(18, 5)
    untouched = reference.observe(18, 0)
    assert moved.window_start_s == untouched.window_start_s
    assert moved.n_pulses == untouched.n_pulses


def test_simulation_feedback(dataset, plan):
    smart = Simulation(dataset, SmartScanScheduler(plan.n_dwells), plan, mode="band_local")
    round_robin = Simulation(dataset, RoundRobinScheduler(plan.n_dwells), plan, mode="band_local")
    for _ in range(600):
        smart.step()
        round_robin.step()
    assert smart.stats["hit_rate"] > round_robin.stats["hit_rate"]
    occupied = [0, 2, 3, *range(5, 13), *range(16, 22)]
    empty = [1, 4, 13, 14, 15, *range(22, 36)]
    probabilities = smart.scheduler.belief.activity_probability
    assert probabilities[occupied].mean() > probabilities[empty].mean()
    visits = smart.history_frame().dwell.value_counts()
    most_visited = visits.sort_values(ascending=False).head(10).index
    assert all(int(dwell) in occupied for dwell in most_visited)


def test_wallclock_round_robin_resonance_artifact(dataset, plan):
    """Wallclock round robin shares the 2.15 s recording sweep period, so it
    phase-locks onto the intervals where the recorded receiver was tuned to each
    band.  Its lead here documents that artifact, not superior scheduling."""
    smart = Simulation(dataset, SmartScanScheduler(plan.n_dwells), plan, mode="wallclock")
    round_robin = Simulation(dataset, RoundRobinScheduler(plan.n_dwells), plan, mode="wallclock")
    for _ in range(300):
        smart.step()
        round_robin.step()
    assert round_robin.stats["hit_rate"] > 4 * smart.stats["hit_rate"]


def test_wasted_steps_counts_only_empty_dwells(dataset, plan):
    simulation = Simulation(dataset, RoundRobinScheduler(plan.n_dwells), plan)
    for _ in range(72):
        simulation.step()
    stats = simulation.stats
    assert stats["wasted_steps_on_empty_dwells"] == 38
    assert stats["misses"] >= stats["wasted_steps_on_empty_dwells"]


def test_reset_restores_seeded_and_learned_state(dataset, plan):
    simulation = Simulation(dataset, RandomScheduler(plan.n_dwells, seed=7), plan)
    first = [simulation.step().dwell for _ in range(10)]
    simulation.reset()
    assert simulation.history == []
    assert [simulation.step().dwell for _ in range(10)] == first
    smart = Simulation(dataset, SmartScanScheduler(plan.n_dwells), plan)
    for _ in range(20):
        smart.step()
    smart.reset()
    assert np.all(smart.scheduler.belief.alpha == 1.0)
