"""EW-SmartScan dashboard.

Streamlit UI over the real TSRD replay simulation. Every number shown here is
produced by ewsmartscan.simulation running against data/config_0.h5 -- there
are no synthetic or random display values.
"""

from __future__ import annotations

import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ewsmartscan.data_loader import DEFAULT_DATA_PATH, load_tsrd
from ewsmartscan.dwells import DwellPlan
from ewsmartscan.schedulers import (
    RandomScheduler,
    RoundRobinScheduler,
    SmartScanScheduler,
    SmartScoreWeights,
)
from ewsmartscan.simulation import Simulation, run_comparison

STEPS_PER_TICK = 5
TICK_SECONDS = 0.25

st.set_page_config(page_title="EW-SmartScan", layout="wide")


@st.cache_resource(show_spinner="Loading TSRD dataset...")
def get_dataset(path: str):
    return load_tsrd(path)


def build_simulation(
    dataset, weights: SmartScoreWeights, horizon: int, mode: str, visit_s: float
) -> Simulation:
    plan = DwellPlan(dataset.receiver)
    scheduler = SmartScanScheduler(plan.n_dwells, weights=weights, coverage_horizon=horizon)
    return Simulation(dataset, scheduler, dwell_plan=plan, mode=mode, visit_duration_s=visit_s)


def reset_state(
    dataset, weights: SmartScoreWeights, horizon: int, mode: str, visit_s: float
) -> None:
    st.session_state.sim = build_simulation(dataset, weights, horizon, mode, visit_s)
    st.session_state.running = False
    st.session_state.mode = mode
    st.session_state.visit_s = visit_s


st.title("EW-SmartScan")
st.caption(
    "Adaptive ESM dwell scheduling on the TSRD dataset: learn where the activity is, "
    "then spend the receiver's time there."
)

dataset = get_dataset(str(DEFAULT_DATA_PATH))
plan = DwellPlan(dataset.receiver)

with st.sidebar:
    st.header("Smart Score weights")
    w_eig = st.slider("w_eig (information gain)", 0.0, 3.0, 1.0, 0.1)
    w_coverage = st.slider("w_coverage (revisit staleness)", 0.0, 3.0, 0.5, 0.1)
    w_activity = st.slider("w_activity (activity probability)", 0.0, 3.0, 2.0, 0.1)
    horizon = st.slider("coverage horizon (steps)", 4, 144, plan.n_dwells, 4)
    st.divider()
    st.header("Replay mode")
    mode = st.radio(
        "mode",
        ["band_local", "wallclock"],
        format_func=lambda m: {
            "band_local": "band_local (default)",
            "wallclock": "wallclock (shows recording artifact)",
        }[m],
        label_visibility="collapsed",
    )
    st.caption(
        "The file is the output of a receiver that swept all 36 bands on a fixed "
        f"{plan.dwell_times_s.sum():.2f} s cycle, so a band only has recorded pulses while the "
        "receiver was tuned to it. band_local replays each band's recorded activity as that "
        "band's own observation timeline (unobserved is not treated as inactive). wallclock "
        "replays a single global clock, which phase-locks a round-robin scheduler to the original "
        "sweep and reproduces its schedule as an artifact."
    )
    visit_s = st.select_slider(
        "dwell budget per decision (s)", options=[0.1, 0.2, 0.4, 0.8, 1.5], value=0.4
    )
    st.divider()
    st.metric("Pulses in dataset", f"{len(dataset.pdws):,}")
    st.metric("Emitters in library", len(dataset.emitters))
    st.metric("Dwell regions", plan.n_dwells)
    st.caption(
        f"Receiver {dataset.receiver.freq_range_mhz[0]:.0f}-"
        f"{dataset.receiver.freq_range_mhz[1]:.0f} MHz, "
        f"{dataset.receiver.bandwidth_mhz:.0f} MHz instantaneous bandwidth, "
        f"{dataset.receiver.collection_time_s:.0f} s collection."
    )

weights = SmartScoreWeights(w_eig=w_eig, w_coverage=w_coverage, w_activity=w_activity)

if (
    "sim" not in st.session_state
    or st.session_state.mode != mode
    or st.session_state.visit_s != visit_s
):
    reset_state(dataset, weights, horizon, mode, visit_s)

sim: Simulation = st.session_state.sim
if sim.scheduler.weights != weights or sim.scheduler.coverage.horizon_steps != horizon:
    sim.scheduler.weights = weights
    sim.scheduler.coverage.horizon_steps = horizon

c_start, c_stop, c_step, c_reset = st.columns(4)
if c_start.button("Start", use_container_width=True, type="primary"):
    st.session_state.running = True
if c_stop.button("Stop", use_container_width=True):
    st.session_state.running = False
if c_step.button("Single step", use_container_width=True):
    st.session_state.running = False
    sim.step()
if c_reset.button("Reset", use_container_width=True):
    reset_state(dataset, weights, horizon, mode, visit_s)
    sim = st.session_state.sim

if st.session_state.running:
    for _ in range(STEPS_PER_TICK):
        sim.step()

scores = sim.scheduler.score_all(sim.step_index)
best = scores.loc[scores["score"].idxmax()]
stats = sim.stats
history = sim.history_frame()

st.subheader(f"CURRENT BEST DWELL: {plan.label(int(best['dwell']))}")
st.write(
    f"**{plan.label(int(best['dwell']))}** covers "
    f"{plan.bounds_mhz(int(best['dwell']))[0]:.0f}-{plan.bounds_mhz(int(best['dwell']))[1]:.0f} MHz. "
    f"Activity probability: **{best['activity_probability']:.3f}** &nbsp;|&nbsp; "
    f"Information Gain: **{best['information_gain']:.3f}** &nbsp;|&nbsp; "
    f"Coverage: **{best['coverage']:.3f}** &nbsp;|&nbsp; "
    f"Final Smart Score: **{best['score']:.3f}**",
    unsafe_allow_html=True,
)
st.latex(
    r"\text{score} = %.1f \cdot \text{IG} + %.1f \cdot \text{coverage} + %.1f \cdot P(\text{activity})"
    % (w_eig, w_coverage, w_activity)
)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Step", stats["steps"])
m2.metric("Receiver time (s)", f"{sim.env.sim_time_s:.2f}")
m3.metric("Hits", stats["hits"])
m4.metric("Hit rate", f"{stats['hit_rate'] * 100:.1f}%")
m5.metric("Emitters detected", stats["unique_emitters_detected"])
m6.metric("Active dwells found", stats["active_dwells_found"])

if sim.history:
    last = sim.history[-1]
    verdict = "HIT" if last.hit else "MISS"
    colour = "#1b7f3b" if last.hit else "#8b1a1a"
    st.markdown(
        f"<div style='padding:0.6rem 1rem;border-radius:0.4rem;background:{colour};color:white;'>"
        f"Step {last.step} &nbsp;→&nbsp; observed <b>{last.dwell_label}</b> at "
        f"t={last.sim_time_s:.3f} s &nbsp;→&nbsp; <b>{verdict}</b> "
        f"({last.n_pulses} pulses, smart score {last.score:.3f})</div>",
        unsafe_allow_html=True,
    )

st.divider()
left, right = st.columns([3, 2])

with left:
    st.subheader("36 dwell regions — score components")
    melted = scores.melt(
        id_vars=["dwell", "label"],
        value_vars=["activity_probability", "information_gain", "coverage"],
        var_name="component",
        value_name="value",
    )
    fig = px.bar(
        melted,
        x="label",
        y="value",
        color="component",
        barmode="group",
        height=340,
        labels={"label": "dwell", "value": ""},
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Smart Score per dwell")
    colours = ["#d62728" if d == int(best["dwell"]) else "#4c78a8" for d in scores["dwell"]]
    fig2 = go.Figure(go.Bar(x=scores["label"], y=scores["score"], marker_color=colours))
    fig2.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="score")
    st.plotly_chart(fig2, use_container_width=True)

with right:
    st.subheader("Dwell table")
    table = scores[
        [
            "label",
            "activity_probability",
            "uncertainty",
            "information_gain",
            "coverage",
            "score",
        ]
    ].copy()
    table.columns = ["Dwell", "P(activity)", "Uncertainty", "Info gain", "Coverage", "Smart score"]
    st.dataframe(
        table.sort_values("Smart score", ascending=False),
        height=430,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Smart score": st.column_config.ProgressColumn(
                "Smart score", format="%.3f", min_value=0.0, max_value=float(table["Smart score"].max())
            ),
            "P(activity)": st.column_config.NumberColumn(format="%.3f"),
            "Uncertainty": st.column_config.NumberColumn(format="%.3f"),
            "Info gain": st.column_config.NumberColumn(format="%.3f"),
            "Coverage": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    st.subheader("Event log")
    if sim.history:
        log = history.tail(15).iloc[::-1]
        log_lines = [
            f"[{int(r.step):04d}] t={r.sim_time_s:6.3f}s  {r.dwell_label}  "
            f"{'HIT ' if r.hit else 'MISS'}  pulses={int(r.n_pulses):3d}  "
            f"score={r.score:.3f}  P(act)={r.activity_probability:.3f}"
            for r in log.itertuples()
        ]
        st.code("\n".join(log_lines), language=None)
    else:
        st.info("Press Start to begin the replay.")

st.divider()
st.subheader("Learning behaviour")
if len(sim.history) >= 2:
    hist = history.copy()
    hist["cumulative_hit_rate"] = hist["hit"].expanding().mean()
    lc, rc = st.columns(2)
    fig3 = px.line(hist, x="step", y="cumulative_hit_rate", height=280)
    fig3.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis_title="cumulative hit rate")
    lc.plotly_chart(fig3, use_container_width=True)

    visits = (
        hist.groupby("dwell_label")
        .agg(visits=("step", "count"), hits=("hit", "sum"))
        .reset_index()
        .sort_values("visits", ascending=False)
        .head(12)
    )
    fig4 = px.bar(visits, x="dwell_label", y=["visits", "hits"], barmode="overlay", height=280)
    fig4.update_layout(
        margin=dict(l=0, r=0, t=10, b=0), legend_title_text="", xaxis_title="dwell"
    )
    rc.plotly_chart(fig4, use_container_width=True)
else:
    st.info("Run a few steps to see the belief converge.")

st.divider()
st.subheader("Baseline comparison")
cmp_steps = st.number_input("Comparison steps", 50, 3000, 400, 50)
if st.button("Run comparison (EW-SmartScan vs Round Robin vs Random)"):
    st.session_state.comparison = run_comparison(
        dataset, int(cmp_steps), weights=weights, mode=mode, visit_duration_s=visit_s
    )
if "comparison" in st.session_state:
    comparison: pd.DataFrame = st.session_state.comparison
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    fig5 = px.bar(comparison, x="scheduler", y="hit_rate", height=280, text_auto=".3f")
    fig5.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis_title="hit rate")
    st.plotly_chart(fig5, use_container_width=True)

if st.session_state.running:
    time.sleep(TICK_SECONDS)
    st.rerun()
