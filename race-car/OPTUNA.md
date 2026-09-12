# Optuna: distance with crash robustness

## Outcome

Ran **80 completed TPE trials**, each on 16 full 3600-tick games, then tested
five distinct finalists plus the original baseline on 50 separate selection
seeds. Trial **62** won the predeclared robust-distance selection rule.
Its parameters were frozen before testing on 100 untouched paired seeds.

| Metric | Previous defaults | Optuna-selected |
|---|---:|---:|
| Finished games | 95/100 | **98/100** |
| Mean distance, including crashes | 148,559.145 | **201,242.078** |
| Median distance | 150,727.050 | **197,864.800** |
| 10th-percentile distance | 67,113.960 | **103,451.710** |
| Best distance | 284,465.400 | **413,091.600** |
| Worst distance | 4,439.200 | **34,120.800** |

Mean-distance improvement: **35.46%**, or **52,682.933** units per game.
The selected configuration beat baseline distance on **75/100** paired seeds.
A seeded 2000-resample paired bootstrap gives a 95% interval for the mean gain
of **[37,398.228, 68,682.734]**.

Crashes fell from five to two in this sample. The selected survival estimate's
95% Wilson interval is **93.00%–99.45%**, versus **88.82%–97.85%** for baseline.
Those intervals overlap: do not claim a statistically established crash-rate
improvement or guaranteed 98% survival. Both observed distance and survival
improved, but there is no all-seed safety proof.

Measured mean decision time during the eight-worker test was 1.558 ms selected
and 2.559 ms baseline. CPU contention varied; these are **not** a controlled
latency comparison and exclude HTTP/network delay. All reported games used
the default sixteen sensors and the unchanged original physics.

The API and demo now load [configs/expert.json](configs/expert.json). The
original `Config()` defaults remain intact for reproducible baseline comparisons.

## Why one million is impossible

The ego starts at speed 10. Acceleration adds at most 0.1 per tick and the game
lasts at most 3600 ticks. Distance accumulates **after** acceleration. Even with
no traffic and `ACCELERATE` on every tick,

$$D_{max}=\sum_{t=1}^{3600}(10+0.1t)
=36,000+0.1\frac{3600\cdot3601}{2}=684,180.$$

This is an absolute upper bound, not a claimed attainable score in traffic.
Steering and braking consume acceleration opportunities. **1,000,000 per game
requires changing the duration or physics**, which this search deliberately
does not do. An aggregate over multiple games is a different metric.

## Objective and selection

[optuna_search.py](optuna_search.py) uses Optuna's TPE sampler. The policy itself
remains a deterministic hand-written controller: no network, learned action
policy, seed lookup, or hidden simulator state.

The scalar objective is

$$U=(0.8\,\overline D+0.2\,Q_{0.1}(D))\,S^2,$$

where $S$ is the fraction of games finishing all 3600 ticks without a crash.
Crashes retain their actual truncated distance and also reduce $S^2$; all-crash
candidates score zero. The lower-tail term discourages relying on a few very
high scores. A distance/survival Pareto set is also exported for inspection.

The shortlist combines high robust-utility trials and high-distance trials
with at least 90% training survival. On the independent selection set, the
predeclared **96% minimum observed survival** takes priority, followed by robust
utility. If no candidate meets that floor, the implementation ranks survival
first and clearly reports the resulting scores. The old baseline is always
evaluated as a fallback. This is an empirical constraint, not a probability
guarantee.

In selection, trial 62 achieved **214,218** mean distance and **48/50** finishes;
the original baseline achieved 149,124.802 and 44/50. Selection-set values are
not the final test results.

### Search space and selected values

| Parameter | Range | Selected |
|---|---|---:|
| Horizon | 90–300, step 15 | 195 |
| Maximum batch | 6, 8, 10, 12, 16, 20 | 8 |
| Collision margin | 4–36 | 6.611122 |
| Future-motion uncertainty | 0.005–0.12, log scale | 0.078802 |
| Terminal-speed weight | 10–240, log scale | 74.665891 |
| Blind-speed history | 40–220, step 10 | 140 |
| Blind-speed buffer | 1–9 | 1.401463 |
| Stale-velocity uncertainty | 0.02–0.22 | 0.170423 |
| Lane-change penalty | 0.01–1, log scale | 0.495474 |

The JSON preserves full precision. Lowering some safety margins while raising
others changes the speed/safety tradeoff; individual values should not be
interpreted in isolation. No controller logic was tuned against this test set.

## Reproduce and extend

From this directory:

```sh
python -m pip install -r requirements-dev.txt
python benchmark.py --config configs/expert.json --games 100 --prefix optuna-v1-test- --output /tmp/optuna-selected.json
python benchmark.py --games 100 --prefix optuna-v1-test- --output /tmp/optuna-baseline.json
python -m pytest tests -q
```

The API/demo use the new preset automatically. `RACE_CAR_CONFIG` can override
the deployment JSON path. Benchmark `--config` accepts either a plain config or
a report with a top-level `config` object; explicit CLI parameter flags override
loaded values. Benchmarking without `--config` uses the previous baseline.

Start a **new** search with disjoint seeds:

```sh
python optuna_search.py --trials 80 --games 16 --workers 8 --validation-games 50 --holdout-games 100 --finalists 5 --prefix optuna-v2- --output results/optuna-v2
```

Use `--phase search`, `--phase select`, and `--phase test` to run phases
separately with the same other settings. `--trials` means total completed trials,
not additional trials. Re-running the search phase resumes its SQLite study
until that total is reached. Only the phase and trial budget may be changed
without altering the study contract; code, dependencies, seeds and selection
settings are checked against stored metadata. Worker count may also vary.

Trials run **sequentially** for stable TPE suggestions; episodes use a process
pool. Threads must not share the simulator's global state or RNG. Each episode
resets its seed. Optuna uses seed 20260912 and 16 startup trials. A fresh,
uninterrupted run is reproducible with the same inputs and versions. Resuming
restarts the sampler RNG, so later suggestions need not exactly match an
uninterrupted run; completed trials and per-config game scores remain reusable.

After the test set has been exposed, further search/selection in that output
directory is blocked. Use a **new output directory and new seed prefix** to
continue optimizing. Do not repeatedly tune to the same holdout failures.
The search writes selected parameters but does not silently deploy them;
the recorded winner was explicitly copied into the deployment preset after
evaluation. Future runs require the same deliberate promotion step.

## Saved evidence

- [results/optuna-v1/trials.json](results/optuna-v1/trials.json): all 80 trials,
  parameters, per-game results, source/asset hashes, Optuna version, seed lists
  and distance/survival Pareto trial numbers.
- [results/optuna-v1/selection.json](results/optuna-v1/selection.json): six
  candidates on selection seeds and the frozen winner.
- [results/optuna-v1/selected-config.json](results/optuna-v1/selected-config.json):
  exported winner, identical to the deployment preset.
- [results/optuna-v1/baseline-config.json](results/optuna-v1/baseline-config.json):
  old configuration, including newly exposed constants at their old values.
- [results/optuna-v1/holdout.json](results/optuna-v1/holdout.json): both policies
  on the same 100 untouched seeds, confidence intervals and paired comparison.

In total, the main run evaluated **1,780 games**: 1,280 search, 300 selection,
and 200 test. SQLite storage is retained locally for inspection/resume but is
Git-ignored; JSON artifacts are portable. Earlier reports under other seed
prefixes remain historical and are not used as this search's test set.

Final verification: **31 tests passed**; source/asset hashes still match the
saved evaluation, and the deployed preset matches the selected JSON exactly.
A full in-process HTTP game with seed `optuna-api-check` matched direct
simulation: 3600 ticks, no crash, distance 102,080.8, 455 requests.

Missing-sensor robustness, network delays and remote competition evaluation
were not revalidated by this search. The single-stream/one-worker API constraint
still applies, and shorter batches may increase network sensitivity.