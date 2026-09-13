"""Paired full-game comparison with isolated worker processes; never deploys."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
import multiprocessing
from pathlib import Path

from expert import load_config
from optuna_search import evaluate, initialize_worker, paired_comparison, save_json, source_hashes
from benchmark import provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs" / "expert.json")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--prefix", default="maneuver-dev-")
    parser.add_argument("--delays", type=int, nargs="+", default=[12, 24, 40])
    parser.add_argument("--recovery-ticks", type=int, default=120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--drift-speeds", type=float, nargs="+",
                        help="Compare drift variants against the loaded base preset instead of legacy/delays")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.games, args.workers) < 1:
        parser.error("games and workers must be positive")
    if args.output.exists():
        parser.error("Output already exists; preserve reports by choosing a new path")
    base = load_config(args.config)
    if args.drift_speeds:
        candidates = [(base.planner, base)] + [(f"drift-speed-{speed}", replace(base, planner="drift",
                       drift_speed=speed)) for speed in args.drift_speeds]
    else:
        base = replace(base, planner="legacy")
        candidates = [("legacy", base)] + [(f"maneuver-delay-{delay}", replace(base, planner="maneuver",
                       maneuver_delay=delay, recovery_ticks=args.recovery_ticks)) for delay in args.delays]
    seeds = [f"{args.prefix}{i}" for i in range(args.games)]
    report = {"provenance": provenance(), "source_hashes": source_hashes(), "seeds": seeds,
              "candidates": []}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker) as pool:
        for label, config in candidates:
            result = {"label": label, **evaluate(pool, seeds, asdict(config))}
            if report["candidates"]:
                result["comparison"] = paired_comparison(result, report["candidates"][0], 20260913)
            report["candidates"].append(result)
            save_json(args.output, report)
            print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    main()