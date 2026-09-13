# Experimental maneuver planner

The first implementation of multi-stage motion planning is available, but it
does **not** replace the deployed Optuna preset automatically. Original road,
vehicle dynamics, randomization, sensors and collision rules are unchanged.
The driver still receives only the public DTO.

**Fresh test result:** 13.4% higher mean distance with the same observed survival,
but substantially higher decision cost. The optional preset is available for
local experiments; the lower-latency Optuna controller remains deployed.

## Implemented

- **115 fixed two-stage candidates**, including all 15 old lane/throttle plans,
  accelerate/coast/brake prefixes followed by lane settling, and short steering
  pulses followed by lane settling with braking. The experimental preset uses
  12- and 24-tick throttle prefixes and up to 8-tick steering pulses.
- **Exact per-tick mixed-action integration.** Velocity changes before movement;
  forward speed clips to zero each tick. Lateral momentum continues during a
  throttle prefix, rather than being reset or silently damped.
- **Batch-end recovery screening.** For each distinct proposed action/count,
  simulate the batch followed by 15 lane/throttle rescue continuations for up
  to 120 additional ticks. Prefix collisions and absolute time/distance offsets
  are included. Predicted recoverable options are preferred; if none exist,
  return the best available valid action rather than an empty response.
- **Adaptive batching near danger**, down to one tick. Batches remain homogeneous
  and stop before planned action changes, retaining FIFO/LIFO compatibility.
- **Remaining-game-time scoring.** Clip the horizon at tick 3600 and decay the
  terminal-speed reward to zero when there is no time beyond the horizon.
- **Braking-reserve correction.** A delayed lane-change plan must preserve
  braking distance while it still overlaps its current lane. A different
  destination no longer exempts its throttle prefix from that check.

This is a bounded motion-primitive search, **not** a general beam search over
arbitrary action strings. Plans are recomputed after each batch; the controller
does not commit to an entire prefix. Additional candidates do not guarantee
better decisions, especially with incomplete traffic observations.

## Architecture and compatibility

[maneuver.py](maneuver.py) implements pure NumPy motion rollouts, candidate
generation, recovery checking and action selection. It imports no simulator or
Pygame state. [expert.py](expert.py) retains observation/tracking and now exposes
a reusable `collision_mask()` for both planners.

`Config.planner` defaults to `legacy`. Existing configuration JSONs load without
changes; the API/demo's [configs/expert.json](configs/expert.json) remains the
Optuna baseline. The new planner can be selected with
[configs/maneuver-experimental.json](configs/maneuver-experimental.json).

Both planners preserve deterministic retry handling, per-game resets and
homogeneous response batches. Source hashes for both planner modules are saved
in benchmark provenance. Historical Optuna studies detect changed source and
must not be resumed under the new code; use a new study for future optimization.

## Run

From the race-car directory:

```sh
python benchmark.py --config configs/maneuver-experimental.json --games 10
python benchmark.py --config configs/maneuver-experimental.json --seeds demo-maneuver --visual
python benchmark.py --config configs/expert.json --games 10
python compare_planners.py --games 10 --prefix new-development- --delays 12 24 40 --workers 4 --output results/new-development.json
python -m pytest tests -q
```

To use the experimental preset in the demo or API, explicitly set
`RACE_CAR_CONFIG` to its path. API deployment still requires one ordered game
stream per worker; shorter requests can be more sensitive to network latency.

The shorthand benchmark flag `--planner maneuver` switches modes but leaves
the default 24-tick prefix setting. Use the experimental JSON to reproduce the
12-tick development candidate exactly.

## Development results

Ten string seeds `maneuver-dev-0` through `maneuver-dev-9` were used for debugging
and choosing prefix duration. They are **not** held-out tests.

| Planner | Finished | Mean distance |
|---|---:|---:|
| Deployed Optuna baseline | 9/10 | 228,678.18 |
| Initial maneuver implementation | 7/10 | 207,408.10 |
| Corrected, delay 12 | 9/10 | 258,030.45 |
| Corrected, delay 24 | 9/10 | 205,454.11 |
| Corrected, delay 40 | 9/10 | 241,730.42 |

The initial version exposed the destination-lane braking-reserve loophole.
Correcting it improved these results, but the sample is small and the candidate
was selected using this data. The new planner is still several times slower
than the baseline. See [results/maneuver-initial.json](results/maneuver-initial.json)
and [results/maneuver-development.json](results/maneuver-development.json).
The former records a previous revision, not the final planner source.

## Frozen 50-seed comparison

The delay-12 candidate was frozen before testing string seeds
`maneuver-test-v1-0` through `maneuver-test-v1-49`. No planner or parameter changes
were made based on these test results. Both controllers used the same seeds and
the unmodified original 3600-tick simulator, with all sixteen sensors.

| Metric | Optuna baseline | Maneuver planner |
|---|---:|---:|
| Finished | 49/50 | 49/50 |
| Mean distance, including crashes | 220,657.662 | **250,286.708** |
| Median distance | 233,229.200 | **251,723.150** |
| 10th-percentile distance | 90,933.690 | **133,513.550** |
| Best distance | 408,606.800 | **426,067.200** |
| Mean decision time | 0.906 ms | 12.571 ms |

Mean improvement: **29,629.046 units, or 13.43%**; new planner wins on 33/50
paired seeds. The 2000-resample paired-bootstrap 95% interval for the mean gain
is **[3,646.350, 56,050.546]**. Both survival estimates have a 95% Wilson interval
of approximately **89.50%–99.65%**. One crash per policy remains; this does not
establish equal safety for arbitrary seeds.

Timing was collected with four simulation workers and variable CPU contention,
so it is not an isolated microbenchmark. Nevertheless the extra computational
cost is substantial. HTTP/network delays and their interaction with short
adaptive batches were not evaluated. For that reason the default deployment
has **not** been promoted to the new planner.

The complete configurations, results and hashes are saved in
[results/maneuver-holdout.json](results/maneuver-holdout.json). To reproduce the
comparison, choose a new output path (the comparison script refuses to overwrite
an existing report):

```sh
python compare_planners.py --games 50 --workers 4 --delays 12 --prefix maneuver-test-v1- --output /tmp/maneuver-reproduction.json
```

For further tuning use a different development prefix, then a **new untouched
test prefix**, rather than tuning against this report's failure seeds.

## Verification and limits

**45 tests passed.** Tests cover candidate coverage; speed clipping across brake/accelerate phases;
lane-change settling and persistent lateral momentum; original simulator motion
parity; recovery-prefix collision rejection, timing offsets and veto behavior;
last-tick scoring; the delayed-lane reserve regression; homogeneous batching;
deterministic replay; API response/retry behavior; and unchanged legacy tests.

The recovery check uses the existing heuristic traffic tracker and uncertainty
model. It is a finite-horizon feasibility screen, **not a proof of safety**.
Unobserved cars, incorrect tracks and maneuvers outside its library can still
cause crashes. It also may reject feasible trajectories that the rescue library
does not contain.

Interval-based tracking, clear-ray invalidation of stale tracks, arbitrary
multi-lane waypoint sequences, planner-specific Optuna optimization, and measured
network-latency handling are not included in this first implementation.