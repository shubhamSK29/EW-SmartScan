"""Run schedulers against the replay environment and collect metrics."""

from dataclasses import dataclass

import pandas as pd

from .data_loader import TsrdDataset
from .dwells import DwellPlan
from .environment import BAND_LOCAL, ReplayEnvironment
from .schedulers import (
    RandomScheduler,
    RoundRobinScheduler,
    SmartScanScheduler,
    SmartScoreWeights,
)


@dataclass
class StepRecord:
    step: int
    sim_time_s: float
    dwell: int
    dwell_label: str
    hit: bool
    n_pulses: int
    score: float
    activity_probability: float
    information_gain: float
    coverage: float
    emitter_ids: tuple[int, ...]


class Simulation:
    """Connect one scheduler to one independent replay environment."""

    def __init__(
        self,
        dataset: TsrdDataset,
        scheduler,
        dwell_plan: DwellPlan | None = None,
        start_time_s: float = 0.0,
        mode: str = BAND_LOCAL,
        visit_duration_s: float | None = 0.4,
    ):
        self.dataset = dataset
        self.dwell_plan = dwell_plan or DwellPlan(dataset.receiver)
        self.scheduler = scheduler
        self.start_time_s = start_time_s
        self.mode = mode
        self.visit_duration_s = visit_duration_s
        self.env = ReplayEnvironment(
            dataset,
            self.dwell_plan,
            start_time_s,
            mode=mode,
            visit_duration_s=visit_duration_s,
        )
        self.environment = self.env
        self._empty_dwells = frozenset(
            int(dwell) for dwell, count in enumerate(self.env.n_pulses_per_dwell()) if count == 0
        )
        self.history: list[StepRecord] = []
        self.step_index = 0

    def step(self) -> StepRecord:
        step_number = self.step_index
        score_data = (
            self.scheduler.score_all(step_number)
            if isinstance(self.scheduler, SmartScanScheduler)
            else None
        )
        dwell = self.scheduler.select(step_number)
        observation = self.env.observe(dwell, step_number)
        self.scheduler.update(observation)
        if score_data is None:
            score = activity = information_gain = coverage = float("nan")
        else:
            selected = score_data.iloc[dwell]
            score = float(selected["score"])
            activity = float(selected["activity_probability"])
            information_gain = float(selected["information_gain"])
            coverage = float(selected["coverage"])
        record = StepRecord(
            step=step_number,
            sim_time_s=observation.sim_time_s,
            dwell=dwell,
            dwell_label=self.dwell_plan.label(dwell),
            hit=observation.hit,
            n_pulses=observation.n_pulses,
            score=score,
            activity_probability=activity,
            information_gain=information_gain,
            coverage=coverage,
            emitter_ids=observation.emitter_ids,
        )
        self.history.append(record)
        self.step_index += 1
        return record

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame([record.__dict__ for record in self.history])

    @property
    def stats(self) -> dict[str, int | float]:
        hits = sum(record.hit for record in self.history)
        wasted = sum(record.dwell in self._empty_dwells for record in self.history)
        emitters = {emitter for record in self.history for emitter in record.emitter_ids}
        dwells = {record.dwell for record in self.history}
        active = {record.dwell for record in self.history if record.hit}
        steps = len(self.history)
        return {
            "steps": steps,
            "hits": hits,
            "hit_rate": hits / steps if steps else 0.0,
            "misses": steps - hits,
            "wasted_steps_on_empty_dwells": wasted,
            "unique_emitters_detected": len(emitters),
            "dwells_visited": len(dwells),
            "active_dwells_found": len(active),
        }

    def reset(self) -> None:
        self.env.reset()
        self.history.clear()
        self.step_index = 0
        reset_scheduler = getattr(self.scheduler, "reset", None)
        if callable(reset_scheduler):
            reset_scheduler()


def run_comparison(
    dataset: TsrdDataset,
    n_steps: int,
    weights: SmartScoreWeights | None = None,
    seed: int = 0,
    mode: str = BAND_LOCAL,
    visit_duration_s: float | None = 0.4,
) -> pd.DataFrame:
    """Run Smart, Random, and Round Robin schedulers independently."""
    plan = DwellPlan(dataset.receiver)
    schedulers = [
        SmartScanScheduler(plan.n_dwells, weights=weights or SmartScoreWeights()),
        RandomScheduler(plan.n_dwells, seed),
        RoundRobinScheduler(plan.n_dwells),
    ]
    rows = []
    for scheduler in schedulers:
        simulation = Simulation(
            dataset, scheduler, plan, mode=mode, visit_duration_s=visit_duration_s
        )
        for _ in range(n_steps):
            simulation.step()
        rows.append({"scheduler": scheduler.name, **simulation.stats})
    return pd.DataFrame(rows)
