# Hand-written race-car expert

An optional [maneuver planner](MANEUVER.md) now adds throttle-first sequences,
batch-end recovery screening and remaining-time scoring. It is available via
[configs/maneuver-experimental.json](configs/maneuver-experimental.json), not
enabled by default; see its separate fresh-seed comparison and latency caveat.

**Update:** the API and demo now load the Optuna-selected preset from
[configs/expert.json](configs/expert.json). On 100 new paired seeds it increased
mean distance by **35.5%**, finishing **98/100** games. See [OPTUNA.md](OPTUNA.md)
for the search, scoring ceiling, and current results. The original benchmark
results below are retained as history, not results for the new preset.

## Run

From this directory, with Python 3.12 and the dependencies in
[requirements-dev.txt](requirements-dev.txt):

```sh
python -m pip install -r requirements-dev.txt
python example.py
python api.py
```

Run the demo and API separately: the demo is a local simulation, not an HTTP
client. The API listens on port 9052. Asset paths are resolved relative to the
project, so invoking the scripts from the repository root also works.

```sh
python benchmark.py --config configs/expert.json --games 10 --output results/local.json
python benchmark.py --config configs/expert.json --seeds my-seed --visual
python benchmark.py --config configs/expert.json --seeds my-seed --trace --output /tmp/race-trace.json
python optimize.py --games 10 --horizons 120 150 180 210 --batches 8 12
python -m pytest tests -q
```

Seed arguments are **strings**: `"565318"` and integer `565318` produce different
random sequences. CLI runs deliberately use string seeds consistently.
Sweeps print the best candidate; they do not silently rewrite the controller.
The API/demo use [settings.py](settings.py) to load the deployed preset. Set
`RACE_CAR_CONFIG` to a different JSON path to override it. The benchmark and
search intentionally retain `Config()` as the **old baseline**; use `--config`
to benchmark the deployed preset.

## What is deterministic?

All original race-car source files and the three sprite assets were inspected.
The road, physics, sensors and collision geometry are deterministic. Traffic
uses a seeded `random.Random` instance, so identical initial state, seed,
sensor configuration and action sequence replay exactly. Changing actions
changes retirement/spawning and eventually the sequence of random draws.

The public request contains only `did_crash`, `elapsed_ticks`, `distance`,
`velocity` and `sensors`. It does **not** expose the seed, other cars, RNG state,
or ego position. Consequently deterministic simulation does not imply perfectly
predictable traffic from these partial observations. The expert does not read
any hidden state or guess seeds.

### Verified geometry

- Screen: 1600 × 1200; road between y=40 and y=1160.
- Five lanes, each 224 pixels high.
- All three sprites scale to 360 × 179 pixels; collisions use the full bounding
  rectangles, not just the painted body of the car.
- Integer lane-center sensor coordinates: 151, 375, 599, 823, 1047.
- Ego center x stays at 800; only its y-position moves on screen.
- Sixteen rays, 22.5 degrees apart, each 1000 pixels long.
- A sensor angle $\theta$ gives direction $(\sin\theta,-\cos\theta)$.
  Thus `STEER_LEFT` moves toward the top wall and `front` points right on screen.

The starter README referred to eight sensors, but initialization enables all
sixteen unless `sensor_removal` is explicitly used.

### Motion equations

An action changes velocity **before** movement. For acceleration and braking,

$$v^x_{t+1}=\max(0,v^x_t+0.1u_x), \qquad D_{t+1}=D_t+v^x_{t+1}.$$

For steering,

$$v^y_{t+1}=v^y_t+0.1u_y, \qquad y_{t+1}=y_t+v^y_{t+1}.$$

Only one action is available per tick: steering and throttle cannot happen
simultaneously. There is no automatic lateral damping and no forward speed cap.
For traffic, relative position changes by $v^x_{car}-v^x_{ego}$, then traffic
velocity changes by a seeded uniform increment in $[-0.1,0.1)$.

Braking lateral motion takes approximately $|v_y|/0.1$ ticks and covers
$v_y|v_y|/0.2$ pixels. This yields the hard-coded bang-bang switching rule:
steer toward the target until the projected stopping position reaches it,
then counter-steer. A four-pixel settling deadband avoids integer sensor
rounding causing endless corrections that starve throttle/braking.

## Controller

[expert.py](expert.py) depends only on NumPy and the standard library. It never
imports Pygame or the simulator. Each decision:

1. **Recover ego y.** Integrate prior actions and constrain the estimate with
   wall distances. Car-blocked rays provide inequalities, not false wall hits.
2. **Reconstruct traffic.** Convert ray distances to hit coordinates and map
   them to known lane rectangles. Front/back face hits determine longitudinal
   position; top/bottom hits give intervals. Keep one track per lane.
3. **Estimate velocity.** Correct changes in relative car position by the ego
   distance travelled. Allow position and velocity uncertainty to grow while
   cars are occluded. Initial velocity estimates use recent ego-speed history.
4. **Account for blind spots.** Inverse ray casting tests possible hidden car
   locations on a 20px grid. A missing sensor key is not a clear ray; a present
   key with `null` is a ray clear to 1000px. Unknown lanes are not simply empty.
5. **Enumerate fixed plans.** Five lane targets × accelerate/coast/brake, with
   explicit tick-by-tick steering and longitudinal motion over a configurable
   horizon (195 ticks in the Optuna preset, 210 in the original baseline).
6. **Rank plans.** Predicted safe duration dominates distance and terminal speed.
   Inflate traffic rectangles for uncertainty, respect wall bounds, and reserve
   braking distance for a possible unseen lead car. Replan after each batch.
7. **Batch safely.** Return up to 8 repeated copies of one action (12 in the
   original baseline), ending the
   batch before any action change. This works for FIFO and LIFO execution;
   the original local loop uses `pop()` and otherwise reverses mixed batches.

This is a small hand-written model-predictive controller, not a learned policy.
The lane targets, dynamics, candidate controls and safety rules are explicit.
Uncertainty multipliers and hidden-car speed estimates are heuristics, not
worst-case proofs. The planner considers only 15 fixed candidates, not every
possible action sequence, so it can miss a feasible maneuver.

## Original benchmark results (before Optuna)

The original controller was frozen before evaluating string seeds `holdout-0` through
`holdout-99`, 3600 ticks each, all sixteen sensors:

| Policy | Finished 3600 ticks | Mean distance | Median distance |
|---|---:|---:|---:|
| Expert | **97/100** | **157,837.499** | **155,699.550** |
| Always accelerate | 0/100 | 7,539.340 | 3,905.550 |
| Coast at initial speed | 4/100 | 10,751.800 | 6,410.000 |

The expert's mean is 20.9× always accelerating and 14.7× coasting. These are
simple local reference policies, **not the competition's unpublished baseline**.
Distance includes crashed runs. Mean decision time was **1.174 ms**, excluding
sensor generation, HTTP, rendering and network delay. Best distance was 266,526;
worst was 24,867.9. These numbers are raw distance, not normalized competition
scores. **Three held-out games crashed.** There is no all-seed safety guarantee.

Raw, per-seed reports, including configuration, package versions and controller
SHA-256:

- [results/final-holdout.json](results/final-holdout.json)
- [results/holdout-accelerate.json](results/holdout-accelerate.json)
- [results/holdout-coast.json](results/holdout-coast.json)

Reproduce those runs:

```sh
python benchmark.py --games 100 --prefix holdout- --output /tmp/expert.json
python benchmark.py --games 100 --prefix holdout- --policy accelerate --output /tmp/accelerate.json
python benchmark.py --games 100 --prefix holdout- --policy coast --output /tmp/coast.json
```

### Optimization history

The ten `tune-*` seeds were used for initial debugging and parameter sweeps.
The fifty `validation-*` seeds were subsequently used to improve safety, so
they are **development data**, not an untouched test set.

| Development stage | Set | Finished | Mean distance |
|---|---|---:|---:|
| Initial geometric planner | 10 tuning seeds | 2/10 | 188,591 |
| Blind-spot handling + steering deadband | 10 tuning seeds | 10/10 | 195,818 |
| Best initial horizon/batch sweep | 10 tuning seeds | 10/10 | 228,352 |
| That candidate on larger development set | 50 validation seeds | 45/50 | 210,495 |
| Final safety rules | 50 validation seeds | 49/50 | 162,917 |

Higher early averages partly reflect aggressive, long runs mixed with early
crashes. Final defaults intentionally trade distance for completion rate.
[results/sweep.json](results/sweep.json) and the other intermediate reports
record earlier controller revisions: current code is **not** expected to
reproduce those intermediate numbers solely by reusing their configuration.
Final reports include a source hash; older development reports do not.

### Validation beyond score

- **20 tests passed**, covering ray intersections, sprite/wall geometry,
  rectangle edge contact, sensor reconstruction, steering settling, predicted
  versus actual motion, original update/RNG parity, every-tick collision
  checking, API schema validation, retry handling, deterministic replay,
  order-independent batches and two full-game regressions.
- Full game through FastAPI's in-process HTTP test client: string seed
  `http-replay`, 3600 ticks, no crash, distance 167,818.2, 309 requests.
- Eight randomly removed sensors: only **6/10** games finished, mean distance
  95,985.42; see [results/sensor-stress.json](results/sensor-stress.json).
  Missing inputs are tolerated, but this controller is tuned for sixteen
  sensors. Sensor removal also consumes RNG draws, so this is a separate
  robustness test, not matched traffic against the sixteen-sensor runs.

## Simulator fidelity and deployment

[benchmark.py](benchmark.py) invokes the original action, car update, retirement,
spawn, and sensor functions. It checks collisions every tick, keeps all RNG
calls (including calls made when the car bucket is empty), and never provides
traffic state to the controller. Optional traces record actual traffic only
**after** making the decision, for offline diagnosis.

The benchmark skips the FPS limiter/rendering and updates sensors only when
requesting actions; sensor updates neither move cars nor consume randomness.
It explicitly checks collisions because the starter's `update_game()` helper
does not. The original `game_loop()` also has an erroneous `MAX_MS` constant;
the benchmark uses the documented 3600-tick limit rather than that wall-clock
constant. Original physics, spawning, geometry and RNG have not been changed.

The repaired [api.py](api.py) returns exactly `{"actions": [...]}`, matching
[dtos.py](dtos.py). The starter incorrectly indexed a list as a dictionary and
referenced a nonexistent `action_type` field. The API no longer imports the
Pygame demo. Invalid numeric inputs return validation errors rather than
breaking the controller. Exact repeated requests reuse the previous response.

**Serve one game stream with one worker.** The competition DTO has no session
identifier, so the server cannot reliably separate interleaved games. A lock
prevents concurrent mutation, but does not make it multi-session. Tick/distance
regression or `did_crash` resets tracking. Do not load-balance a single game
across independent workers. A new session-ID contract is needed for that.

Network latency, repeated last actions during delays, external evaluator
differences, and real remote validation remain untested. Only the local
implementation and in-process HTTP route were exercised; no competition
validation/evaluation attempt was submitted.