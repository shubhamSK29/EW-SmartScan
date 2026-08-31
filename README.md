# EW-SmartScan

A small, working demonstration of an adaptive ESM dwell-scheduling strategy on the
TSRD dataset in `data/config_0.h5`.

> Instead of blindly scanning all frequency bands, the software learns from previous
> observations and intelligently chooses which dwell should be observed next.

```
TSRD dataset -> 36 dwell regions -> activity belief + uncertainty
             -> information gain + coverage -> smart score
             -> best next dwell -> observation -> belief update -> next decision
```

## What is actually in `data/config_0.h5`

Determined by inspection, not assumption:

| Path | Shape / type | Meaning |
| --- | --- | --- |
| `data` | `(169617, 5)` float32 | PDWs |
| `labels` | `(169617, 1)` int8 | emitter id per pulse (0..95) |
| `metadata/feature_names` | `(5,)` bytes | `ToA, Frequency, PulseWidth, AoA, Amplitude` |
| `metadata` attrs | | `collection_time_s=30.0`, `num_pulses=169617`, `type='synthetic'` |
| `metadata/receiver` attrs | | `bandwith_mhz=500.0` (sic), `sensitivity_dbm=-110`, `gain_db=10`, `scan_mode='Scanning'`, noise scales |
| `metadata/receiver/dwell_centres_mhz` | `(36,)` | 250, 750, ... 17750 MHz (500 MHz spacing) |
| `metadata/receiver/dwell_times_s` | `(36,)` | 0.05 s or 0.1 s per dwell |
| `metadata/receiver/freq_range_mhz` | `(2,)` | `[500, 18000]` |
| `metadata/transmitters/transmitters_<0..95>` | groups | `function` name plus frequency / PRI / pulse-width / scan / power / position config |

Units: ToA is microseconds (0.2 s .. 29.2 s), Frequency MHz (10 .. 11000), PulseWidth µs,
AoA degrees, Amplitude dBm. Rows are already sorted by ToA.

Mapping every pulse to its nearest dwell centre gives 17 occupied bands and 19 empty ones:

```
occupied: D00 D02 D03 D05 D06 D07 D08 D09 D10 D11 D12 D16 D17 D18 D19 D20 D21
empty:    D01 D04 D13 D14 D15 D22..D35
```

The 19 empty bands are empty because no emitter in the library radiates there within the
receiver's tuning range (the 35/77/94 GHz emitters are out of band entirely).

## How to run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python run_demo.py --steps 600      # console demonstration
.venv/bin/streamlit run app.py                # dashboard on http://localhost:8501
.venv/bin/python -m pytest tests -q           # 19 tests against the real HDF5 file
```

## Files

| File | Purpose |
| --- | --- |
| `ewsmartscan/data_loader.py` | reads the HDF5 into a typed PDW dataframe + receiver/emitter metadata |
| `ewsmartscan/dwells.py` | the 36 dwell regions and nearest-centre frequency mapping |
| `ewsmartscan/environment.py` | leakage-free replay of the real pulses, one dwell at a time |
| `ewsmartscan/beliefs.py` | Beta-Bernoulli activity belief, entropy, expected information gain |
| `ewsmartscan/coverage.py` | how long since each dwell was last observed |
| `ewsmartscan/schedulers.py` | EW-SmartScan scheduler + Random and Round Robin baselines |
| `ewsmartscan/simulation.py` | the decision loop and the scheduler comparison |
| `run_demo.py` | console demonstration |
| `app.py` | Streamlit dashboard driven by the live simulation |
| `tests/test_ewsmartscan.py` | tests for the maths, the environment and the feedback loop |

## How the decision-making works

Every simulation step is one scheduling decision.

**1. Belief.** Each dwell `d` has a Beta-Bernoulli belief over "this band produces
detections when I look at it", starting from `Beta(1, 1)`:

```
HIT  -> alpha += 1
MISS -> beta  += 1
P(activity) = alpha / (alpha + beta)
variance    = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))
uncertainty = sqrt(variance)
```

**2. Information gain.** The expected reduction in predictive (Bernoulli) entropy from
one more look, in bits:

```
H(p)  = -p*log2(p) - (1-p)*log2(1-p)
p_hit  = (alpha+1) / (alpha+beta+1)
p_miss =  alpha    / (alpha+beta+1)
EIG(d) = H(p) - [ p*H(p_hit) + (1-p)*H(p_miss) ]
```

`p = p*p_hit + (1-p)*p_miss`, so by Jensen's inequality on the concave `H` this is always
`>= 0`, and it shrinks as evidence accumulates. It is reported normalised by its value at
the prior (`EIG_PRIOR ≈ 0.0817` bits, the most any single look can be worth).

**3. Coverage.** `staleness(d) = steps since d was last observed`, and
`coverage(d) = min(1, staleness / horizon)` with the horizon defaulting to 36 steps (one
full sweep). A never-observed dwell scores 1.0.

**4. Smart score and decision.**

```
score(d) = w_eig * information_gain(d) + w_coverage * coverage(d) + w_activity * P(activity|d)
```

with configurable weights (defaults `1.0 / 0.5 / 2.0`). The scheduler picks
`argmax score`, observes only that dwell, updates its belief and coverage from the real
observation, and rescores. Nothing is random and no future data is visible: the
environment only ever reads the chosen dwell's next observation window.

Early on, all dwells share `P=0.5`, `IG=1.0`, `coverage=1.0`, so the tie-break sweeps the
band and explores. As evidence arrives, the empty bands fall to `P -> 0` with `IG -> 0`,
and the active bands rise, so the activity term dominates and the receiver concentrates
its time there — while the coverage term keeps pulling it back to re-check stale bands.

## Results (real data, 600 decisions, default weights)

```
   scheduler  steps  hits  hit_rate  wasted_steps_on_empty_dwells  unique_emitters_detected
EW-SmartScan    600    95     0.158                           176                        55
      Random    600    49     0.082                           338                        42
 Round Robin    600    51     0.085                           311                        38
```

The learned probabilities track the measured occupancy of each band closely:

```
dwell  learned  ground truth        dwell  learned  ground truth
  D17    0.327         0.297          D19    0.270         0.270
  D06    0.302         0.284          D02    0.263         0.257
  D18    0.298         0.297          D11    0.241         0.230
  D05    0.293         0.270          D00    0.240         0.189
```

## Assumptions

1. **Dwell bounds.** `dwell_centres_mhz` gives centres only, so each dwell is taken as
   `centre ± 250 MHz` (half the 500 MHz spacing, which also matches the receiver's
   500 MHz instantaneous bandwidth). This puts D00 at 0–500 MHz, slightly below the
   stated `freq_range_mhz` lower bound of 500 MHz; the 9299 pulses between 10 and
   500 MHz (VHF/UHF and OTH emitters) are assigned there rather than discarded.
2. **HIT definition.** A HIT is "at least one pulse recorded in this band during the
   visit". No detection threshold is applied on top of the data — the file is already
   receiver-limited at `sensitivity_dbm = -110`. `ReplayEnvironment(sensitivity_dbm=...)`
   can impose one if wanted.
3. **The file is a receiver recording, not spectrum ground truth.** The sum of
   `dwell_times_s` is 2.15 s, and every pulse's ToA modulo 2.15 s falls inside its own
   band's scheduled window in that sweep — i.e. the collection was produced by a
   receiver that swept all 36 bands on a fixed 2.15 s cycle, so a band only has recorded
   pulses for the intervals when that receiver was tuned to it. A band being silent at
   any other instant is a *recording gap*, not evidence of inactivity.

   Consequently the default replay mode is `band_local`: each band's recorded activity is
   replayed as that band's own observation timeline, with an independent playback pointer
   per band. The alternative `wallclock` mode (single global clock indexing into the
   collection) is kept only to demonstrate the artifact — there, a Round Robin scheduler
   shares the recording's own 2.15 s period, phase-locks onto the recorded windows and
   scores 0.34 against EW-SmartScan's 0.06. That is resonance with the recording
   schedule, not better scheduling; at other start offsets it collapses to zero.
4. **Decision granularity.** One decision allocates a dwell budget of
   `visit_duration_s = 0.4 s` on the chosen band, i.e. `round(0.4 / dwell_time)`
   consecutive receiver dwell windows. The receiver's dwell length is unchanged; this is
   only how much on-band time one scheduling decision buys, and it is identical for every
   band so scheduler comparisons stay fair. It matters because the emitters are scanning
   radars: a single 50–100 ms window catches an active band only 3–16 % of the time,
   which is too weak to learn from in a few hundred decisions, while a 0.4 s budget
   separates active bands (0.11–0.29) from empty ones (0.00) cleanly.
5. **Emitter identity** is used only for reporting how many distinct emitters were
   detected. The scheduler never sees labels.
6. **Replay wrap-around.** Runs longer than the collection wrap back to the start of each
   band's timeline, so long runs re-observe the same real data rather than inventing any.

## Not included

This is a core-concept demonstrator: no FastAPI, React, WebSockets, MongoDB, SDR or RF
hardware, deep learning, complex DSP, authentication, or deployment infrastructure.
