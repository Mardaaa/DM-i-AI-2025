"""Small, reproducible parameter sweep; no machine learning or seed lookup."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from benchmark import provenance, run_episode, summarize
from expert import Config
import pygame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--prefix", default="tune-")
    parser.add_argument("--horizons", nargs="+", type=int, default=[120, 150, 180, 210])
    parser.add_argument("--batches", nargs="+", type=int, default=[8])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "sweep.json")
    args = parser.parse_args()
    reports = []
    pygame.init()
    try:
        for horizon in args.horizons:
            for batch in args.batches:
                config = Config(horizon=horizon, batch_size=batch)
                results = [run_episode(f"{args.prefix}{i}", config=config) for i in range(args.games)]
                report = {"provenance": provenance(), "config": asdict(config), "summary": summarize(results), "results": results}
                reports.append(report)
                print(json.dumps({k: v for k, v in report.items() if k != "results"}), flush=True)
    finally:
        pygame.quit()
    # Prefer finishing games, then distance. This is selection, not a guarantee.
    reports.sort(key=lambda r: (r["summary"]["survived"], r["summary"]["mean_distance"]), reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + "\n")
    print("Best:", json.dumps(reports[0]["config"]))


if __name__ == "__main__":
    main()