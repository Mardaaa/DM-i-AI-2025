"""Optuna TPE parameter optimization with crash penalties and fresh seed splits.

Trials are sequential for reproducible suggestions. Games run in separate
processes: threads would corrupt the original simulator's global state/RNG.
No search output modifies physics, controller source, or deployed defaults.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import optuna
from optuna.trial import TrialState
import pygame

from benchmark import provenance, run_episode, summarize
from expert import Config


MAX_DISTANCE = 10 * 3600 + 0.1 * 3600 * 3601 / 2
SEARCH_SPACE_VERSION = 1


def initialize_worker():
    pygame.init()


def evaluate_job(job):
    seed, parameters = job
    return run_episode(seed, config=Config(**parameters))


def lower_quantile(values, fraction=0.1):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def wilson_interval(successes, total):
    """95% Wilson interval: observed success rate is not a guarantee."""
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def metrics(results):
    report = summarize(results)
    survival = report["survived"] / len(results)
    p10 = lower_quantile([r["distance"] for r in results])
    # Crashes already truncate distance; additionally penalize fragility and
    # reward the lower tail rather than a few spectacular lucky runs.
    utility = (0.8 * report["mean_distance"] + 0.2 * p10) * survival ** 2
    report.update(survival_rate=survival, p10_distance=p10, robust_utility=utility,
                  survival_95ci=wilson_interval(report["survived"], len(results)))
    return report


def evaluate(pool, seeds, parameters):
    results = list(pool.map(evaluate_job, [(seed, parameters) for seed in seeds]))
    return {"config": parameters, "summary": metrics(results), "results": results}


def sample_config(trial):
    return Config(
        horizon=trial.suggest_int("horizon", 90, 300, step=15),
        batch_size=trial.suggest_categorical("batch_size", [6, 8, 10, 12, 16, 20]),
        margin=trial.suggest_float("margin", 4.0, 36.0),
        uncertainty=trial.suggest_float("uncertainty", 0.005, 0.12, log=True),
        terminal_speed_weight=trial.suggest_float("terminal_speed_weight", 10.0, 240.0, log=True),
        blind_history=trial.suggest_int("blind_history", 40, 220, step=10),
        blind_speed_buffer=trial.suggest_float("blind_speed_buffer", 1.0, 9.0),
        stale_velocity_uncertainty=trial.suggest_float("stale_velocity_uncertainty", 0.02, 0.22),
        lane_change_penalty=trial.suggest_float("lane_change_penalty", 0.01, 1.0, log=True),
    )


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def source_hashes():
    root = Path(__file__).parent
    paths = [root / name for name in ("expert.py", "benchmark.py", "optuna_search.py")]
    paths += sorted((root / "src").rglob("*.py"))
    paths += sorted((root / "public" / "assets").glob("*.png"))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def complete_trials(study):
    return [t for t in study.trials if t.state == TrialState.COMPLETE]


def export_trials(study, metadata, directory):
    trials = complete_trials(study)
    records = [{"trial": t.number, "value": t.value, **t.user_attrs} for t in trials]
    pareto = []
    for trial in records:
        s = trial["summary"]
        dominated = any(
            other["summary"]["mean_distance"] >= s["mean_distance"]
            and other["summary"]["survival_rate"] >= s["survival_rate"]
            and (other["summary"]["mean_distance"] > s["mean_distance"]
                 or other["summary"]["survival_rate"] > s["survival_rate"])
            for other in records
        )
        if not dominated:
            pareto.append(trial["trial"])
    save_json(directory / "trials.json", {"metadata": metadata, "trials": records,
                                         "pareto_distance_survival": pareto})


def shortlist(study, count):
    trials = complete_trials(study)
    # Include best robust utility and best distance among high-survival trials.
    by_utility = sorted(trials, key=lambda t: t.value, reverse=True)
    safe = sorted((t for t in trials if t.user_attrs["summary"]["survival_rate"] >= 0.9),
                  key=lambda t: t.user_attrs["summary"]["mean_distance"], reverse=True)
    ordered = []
    for candidates in zip(by_utility, safe + by_utility):
        for t in candidates:
            if all(t.user_attrs["config"] != old.user_attrs["config"] for old in ordered):
                ordered.append(t)
            if len(ordered) == count:
                return ordered
    return ordered


def selection_key(report, minimum_survival):
    s = report["summary"]
    feasible = s["survival_rate"] >= minimum_survival
    # For feasible candidates optimize robust distance, otherwise safety first.
    return (feasible, s["robust_utility"] if feasible else s["survival_rate"], s["mean_distance"])


def paired_comparison(selected, baseline, seed):
    differences = [a["distance"] - b["distance"] for a, b in zip(selected["results"], baseline["results"])]
    import random
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(2000))
    return {"mean_distance_difference": statistics.mean(differences),
            "paired_bootstrap_95ci": [means[50], means[1949]],
            "distance_ratio": selected["summary"]["mean_distance"] / baseline["summary"]["mean_distance"],
            "seed_wins": sum(d > 0 for d in differences),
            "survival_difference": selected["summary"]["survival_rate"] - baseline["summary"]["survival_rate"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=64, help="Total completed-trial budget, including resumed trials")
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--prefix", default="optuna-v1-")
    parser.add_argument("--finalists", type=int, default=5)
    parser.add_argument("--validation-games", type=int, default=50)
    parser.add_argument("--holdout-games", type=int, default=100)
    parser.add_argument("--minimum-survival", type=float, default=0.96)
    parser.add_argument("--phase", choices=("all", "search", "select", "test"), default="all")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "optuna-v1")
    args = parser.parse_args()
    if min(args.trials, args.games, args.workers, args.finalists, args.validation_games, args.holdout_games) < 1:
        parser.error("Trial/game/worker/finalist counts must be positive")
    if not 0 <= args.minimum_survival <= 1:
        parser.error("minimum-survival must be from 0 to 1")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "holdout.json").exists() and args.phase != "test":
        parser.error("This run's holdout is already exposed. Use a new output AND seed prefix for further tuning.")
    seeds = {split: [f"{args.prefix}{split}-{i}" for i in range(count)] for split, count in
             (("train", args.games), ("select", args.validation_games), ("test", args.holdout_games))}
    metadata = {"source_hashes": source_hashes(), "runtime": provenance(), "optuna": optuna.__version__,
                "search_space_version": SEARCH_SPACE_VERSION, "sampler_seed": args.seed,
                "seeds": seeds, "minimum_survival": args.minimum_survival, "finalists": args.finalists,
                "objective": "(0.8*mean_distance + 0.2*p10_distance) * survival_rate**2",
                "theoretical_max_distance": MAX_DISTANCE}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name="expert-distance-robustness", direction="maximize",
                               storage=f"sqlite:///{directory / 'study.sqlite3'}", load_if_exists=True,
                               sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=16))
    old = study.user_attrs.get("metadata")
    if old is not None and old != metadata:
        parser.error("Existing study has different code/environment/settings/seeds. Use a new output directory.")
    study.set_user_attr("metadata", metadata)
    save_json(directory / "baseline-config.json", asdict(Config()))
    if not study.trials and args.phase in ("all", "search"):
        study.enqueue_trial(asdict(Config()))

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker) as pool:
        if args.phase in ("all", "search"):
            def objective(trial):
                parameters = asdict(sample_config(trial))
                report = evaluate(pool, seeds["train"], parameters)
                for key, value in report.items():
                    trial.set_user_attr(key, value)
                return report["summary"]["robust_utility"]

            def progress(current_study, trial):
                print(json.dumps({"trial": trial.number, **trial.user_attrs["summary"]}), flush=True)
                export_trials(current_study, metadata, directory)

            remaining = max(0, args.trials - len(complete_trials(study)))
            study.optimize(objective, n_trials=remaining, callbacks=[progress])
            export_trials(study, metadata, directory)

        if args.phase in ("all", "select"):
            trials = shortlist(study, args.finalists)
            if not trials:
                parser.error("No completed trials to select; run --phase search first")
            candidates = [("baseline", asdict(Config()))]
            for trial in trials:
                parameters = trial.user_attrs["config"]
                if all(parameters != p for _, p in candidates):
                    candidates.append((f"trial-{trial.number}", parameters))
            reports = []
            for label, parameters in candidates:
                report = {"label": label, **evaluate(pool, seeds["select"], parameters)}
                reports.append(report)
                print("SELECTION", json.dumps({"label": label, **report["summary"]}), flush=True)
            best = max(reports, key=lambda r: selection_key(r, args.minimum_survival))
            save_json(directory / "selection.json", {"metadata": metadata, "selected": best["label"], "candidates": reports})
            save_json(directory / "selected-config.json", best["config"])

        if args.phase in ("all", "test"):
            path = directory / "selection.json"
            if not path.exists():
                parser.error("No selection report; run --phase select first")
            selection = json.loads(path.read_text())
            if selection["metadata"] != metadata:
                parser.error("Selection metadata does not match this evaluation")
            parameters = next(r["config"] for r in selection["candidates"] if r["label"] == selection["selected"])
            selected = evaluate(pool, seeds["test"], parameters)
            baseline = evaluate(pool, seeds["test"], asdict(Config()))
            comparison = paired_comparison(selected, baseline, args.seed)
            report = {"metadata": metadata, "selected_label": selection["selected"], "selected": selected,
                      "baseline": baseline, "comparison": comparison}
            save_json(directory / "holdout.json", report)
            print("HOLDOUT", json.dumps({"selected": selected["summary"], "baseline": baseline["summary"],
                                         "comparison": comparison}), flush=True)


if __name__ == "__main__":
    main()