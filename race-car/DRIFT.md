# First implementation toward 500k: diagnostics and drifting

**Research only. Not a deployment recommendation.** The first drift variants
raise development-set mean distance but crash more often. No 500k run was
achieved. The API/demo continue to use [configs/expert.json](configs/expert.json).
The previous [maneuver preset](configs/maneuver-experimental.json) is unchanged.

## Implemented

- [distance_diagnostics.py](distance_diagnostics.py): passive, per-tick distance
  accounting, action counts, early/middle/late-game breakdown and decision-level
  recovery/target-switch counters. Always recorded by [benchmark.py](benchmark.py).
- [drift.py](drift.py): steering pulse, throttle while lateral momentum persists,
  counter-steering near the target. Three lateral cruise speeds and off-center
  targets add 135 candidates to the original 115, for **250 total**.
- Recovery evaluates the existing 15 continuations plus 135 drift continuations,
  retaining the executed prefix and its absolute observation-time coordinates.
- Identical traffic/wall collision checks, current-lane blind-car reserve and
  homogeneous action batches. No simulator state or RNG is read by the policy.
- Per-episode p95/p99 decision times and summary 500k success count/rate and p10
  distance. Crashes do not count as successful 500k games. Percentiles measure
  local controller calls, not HTTP round-trip time.

The new planner is selected explicitly with `planner="drift"`. `drift_speed`
sets the middle cruise speed; candidates use 0.5x, 1x and 1.5x (rounded to 0.1).
Targets are lane centers and offsets of ±24px, clipped to the existing safe
road-center bounds. These are samples of the lateral corridor, not a continuous
interval optimizer and not an assumption that lane boundaries are safe gaps.

No beam search, interval-state tracker, oracle controller, or new Optuna search
is included in this increment.

## Distance accounting

For initial forward speed $v_0$, tick budget $N$, and executed tick $t$ (1-based):

$$
D_{ceiling}=v_0N+0.05N(N+1),\qquad
L_t=(0.1-\Delta v_t)(N-t+1).
$$

Actual speed deltas include clipping at zero. With $T$ executed ticks,
$R=N-T$ remaining ticks, and final speed $v_T$, termination loss is
$L_{termination}=v_TR+0.05R(R+1)$.

Thus $D_{ceiling}-D_{actual}=\sum L_t+L_{termination}$, up to rounding.
The report includes unrounded values and the identity residual. At 3600 ticks
and initial speed 10 the ceiling is 684,180. An early termination is separated
from the opportunity cost of its executed actions.

**This is accounting, not proof the distance was safely achievable.** Changing
actions changes traffic-relative positions and subsequent RNG consumption; it
does not produce the same traffic counterfactual. Steering/braking can be
necessary. `target_switch_decisions` counts target changes, not completed lane
changes or reversals. Recovery counters count decisions/distinct checked batches,
not safety proofs or per-tick explanations. Observed-versus-unseen rejection
attribution is not yet emitted by the planner.

## Development experiment

Twelve paired string seeds `drift-dev-v1-0` through `drift-dev-v1-11`, original
3600-tick game and sixteen sensors. Both variants reused the maneuver preset's
parameters, adding only the drift library with the specified speed. This is
development data, **not an untouched holdout**.

| Policy | Finished | Mean distance | p10 distance | Best | Mean decision ms |
|---|---:|---:|---:|---:|---:|
| Existing maneuver | 11/12 | 175,020 | 113,561 | 279,497 | 7.979 |
| Drift speed 1 | 8/12 | 200,467 | 11,578 | 371,723 | 38.071 |
| Drift speed 2 | 9/12 | 215,544 | 4,422 | 428,696 | 44.531 |

All three have **0/12 successful 500k games**. Speed 2 improves mean distance
23.15%, but its paired-bootstrap 95% mean-gain interval is approximately
[-43,440, 111,577]; it does not establish a reliable improvement. Survival and
lower-tail performance regress, so neither drift variant qualifies for promotion
or final holdout evaluation. Four process workers contend for CPU; the timing
numbers are not controlled latency measurements and exclude network delay.

Complete configurations, per-game diagnostics, comparisons and source hashes:
[results/drift-development-v1.json](results/drift-development-v1.json).

### What the diagnostics revealed

Mean distance-loss attribution across these twelve games (rounded):

| Loss | Existing maneuver | Drift speed 2 |
|---|---:|---:|
| Steering | 84,160 | 127,066 |
| Coasting | 219,468 | 68,674 |
| Braking | 152,585 | 134,269 |
| Termination | 52,948 | 138,627 |

Despite fewer steering ticks in a constructed single-crossing test, complete
games use **more steering**: about 711 versus 477 ticks/game. Coasting falls
from about 1181 to 388 ticks/game. This cautions against extrapolating an isolated
motion improvement to closed-loop driving. Accounting residuals are below 6e-9.

A trace of development seed 4 confirms the speed-2 car collides during a lane
change with a rear car. By tick 272 all checked batches lack a safe recovery;
the collision follows at tick 288. This establishes late detection of an
unrecoverable situation in that trace, not the root cause or a universal fix.

## Usage

From the race-car directory, use the explicitly development-only preset:

```sh
python benchmark.py --config configs/drift-development.json --games 4 --prefix drift-local- --output /tmp/drift-local.json
```

Reproduce the paired experiment using a new output path (existing comparison
reports cannot be overwritten):

```sh
python compare_planners.py --config configs/maneuver-experimental.json --games 12 --workers 4 --drift-speeds 1 2 --prefix drift-dev-v1- --output /tmp/drift-comparison.json
```

Use different development seeds for broader exploration and a new untouched test
prefix only after a candidate meets the development survival requirement. Do not
tune against previous Optuna or maneuver holdouts. Historical reports' source
hashes refer to their original code revision, not the current files.

## Verification and next work

**84 tests pass**, including loss identity with clipping and early termination,
passive diagnostics preserving observations/actions/RNG, drift acceleration and
settling, original-game motion parity, recovery prefix timing, blind-car reserve,
deterministic replay and API/retry behavior. The existing two third-party
deprecation warnings remain.

Next priorities: trace the first loss of recoverability rather than the final
crash; distinguish observer error from limited escape paths; add multi-stage
abort/retreat planning and test target-switch costs. The development data do not
support simply loosening margins or deploying the higher-mean variant. Continuous
500k performance remains an unachieved research target.