"""Command-line demonstration of the EW-SmartScan scheduler."""

import argparse

from ewsmartscan.data_loader import DEFAULT_DATA_PATH, load_tsrd
from ewsmartscan.dwells import DwellPlan
from ewsmartscan.simulation import Simulation, run_comparison
from ewsmartscan.schedulers import SmartScanScheduler, SmartScoreWeights
from ewsmartscan.environment import ReplayEnvironment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--visit-duration", type=float, default=0.4)
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--w-eig", type=float, default=1.0)
    parser.add_argument("--w-coverage", type=float, default=0.5)
    parser.add_argument("--w-activity", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = load_tsrd(args.data)
    plan = DwellPlan(dataset.receiver)
    weights = SmartScoreWeights(args.w_eig, args.w_coverage, args.w_activity)
    smart = SmartScanScheduler(plan.n_dwells, weights=weights)
    simulation = Simulation(
        dataset, smart, plan, mode="band_local", visit_duration_s=args.visit_duration
    )
    for _ in range(args.steps):
        simulation.step()

    print(
        f"Dataset: {len(dataset.pdws)} pulses, {len(dataset.emitters)} emitters, "
        f"{dataset.receiver.collection_time_s:.1f}s collection"
    )
    print("Dwell plan:")
    for dwell in range(plan.n_dwells):
        lo, hi = plan.bounds_mhz(dwell)
        print(f"  {plan.label(dwell)}: {lo:.0f}-{hi:.0f} MHz ({plan.dwell_times_s[dwell]:.3f}s)")

    print("\nFirst smart-scan decisions:")
    print("step dwell result score eig coverage activity")
    for record in simulation.history[:20]:
        print(
            f"{record.step:4d} {record.dwell_label:>5} {'HIT' if record.hit else 'MISS':>4} "
            f"{record.score:5.3f} {record.information_gain:4.2f} "
            f"{record.coverage:4.2f} {record.activity_probability:4.2f}"
        )

    truth = ReplayEnvironment(
        dataset, plan, visit_duration_s=args.visit_duration
    ).ground_truth_occupancy()
    probabilities = smart.belief.activity_probability
    top = sorted(range(plan.n_dwells), key=lambda i: (-probabilities[i], i))[:10]
    print("\nTop-10 learned activity probabilities vs ground-truth occupancy:")
    print("dwell learned ground_truth")
    for dwell in top:
        print(f"{plan.label(dwell):>5} {probabilities[dwell]:7.3f} {truth[dwell]:12.3f}")

    print(f"\nScheduler comparison (band_local replay, {args.visit_duration:.2f}s visits):")
    print(
        run_comparison(
            dataset, args.steps, weights, args.seed, visit_duration_s=args.visit_duration
        ).to_string(index=False)
    )

    sweep_period_s = float(plan.dwell_times_s.sum())
    print("\n=== Replay mode comparison: band_local vs wallclock ===")
    print(
        f"The collection was recorded by a receiver sweeping all {plan.n_dwells} bands every "
        f"{sweep_period_s:.4f} s, so in wallclock replay a round-robin scheduler shares that "
        "period and phase-locks onto the recorded dwell windows."
    )
    print(
        "Its high wallclock hit rate is that recording artifact, not better scheduling; "
        "band_local replay gives each band its own timeline and asks the real question."
    )
    for mode in ("band_local", "wallclock"):
        print(f"\nmode={mode}")
        print(
            run_comparison(
                dataset,
                args.steps,
                weights,
                args.seed,
                mode=mode,
                visit_duration_s=args.visit_duration,
            ).to_string(index=False)
        )


if __name__ == "__main__":
    main()
