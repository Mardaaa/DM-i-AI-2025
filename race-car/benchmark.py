"""Reproducible headless benchmark using the ORIGINAL game physics and RNG.

The driver sees only DTO fields. Rendering and redundant sensor updates are
skipped, but collision checks, spawning, and all random calls run every tick.
"""

import argparse
from collections import deque
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from expert import Config, ExpertController, load_config
from src.game import core


def observation(state):
    for sensor in state.sensors:
        sensor.update()
    return {"did_crash": state.crashed, "elapsed_ticks": state.ticks,
            "distance": state.distance, "velocity": {"x": state.ego.velocity.x, "y": state.ego.velocity.y},
            "sensors": {sensor.name: sensor.reading for sensor in state.sensors}}


def step(action):
    """Exactly the tick ordering in core.game_loop, without its FPS limiter."""
    state = core.STATE
    state.ticks += 1
    core.handle_action(action)
    state.distance += state.ego.velocity.x
    core.update_cars()
    core.remove_passed_cars()
    core.place_car()
    state.crashed = any(state.ego.rect.colliderect(car.rect) for car in state.cars if car is not state.ego)
    state.crashed |= any(state.ego.rect.colliderect(wall.rect) for wall in state.road.walls)
    return state


def render(screen, state):
    screen.blit(state.road.surface, (0, 0))
    for wall in state.road.walls:
        wall.draw(screen)
    for car in state.cars:
        screen.blit(car.sprite, (car.x, car.y))
    pygame.display.flip()


def run_episode(seed: str, policy="expert", config: Config | None = None,
                max_ticks=3600, visual=False, trace=False, sensor_removal=0):
    with redirect_stdout(io.StringIO()):
        core.initialize_game_state("", seed, sensor_removal=sensor_removal)
    driver = ExpertController(config)
    queue = deque()
    latencies, history = [], []
    screen = pygame.display.set_mode((core.SCREEN_WIDTH, core.SCREEN_HEIGHT)) if visual else None
    clock = pygame.time.Clock() if visual else None
    state = core.STATE
    while state.ticks < max_ticks and not state.crashed:
        if visual:
            if any(event.type == pygame.QUIT for event in pygame.event.get()):
                break
            clock.tick(60)
        if not queue:
            obs = observation(state)
            start = time.perf_counter()
            if policy == "expert":
                queue.extend(driver.actions(obs))
            else:
                queue.extend(["ACCELERATE" if policy == "accelerate" else "NOTHING"] * 8)
            latencies.append((time.perf_counter() - start) * 1000)
            if trace:
                history.append({**obs, "plan": dict(driver.last_plan), "actions": list(queue),
                                "actual_y": state.ego.rect.centery,
                                "traffic": [{"lane": core.STATE.road.lanes.index(c.lane), "x": c.rect.centerx - 800,
                                             "speed": c.velocity.x} for c in state.cars if c is not state.ego],
                                "tracks": {lane: asdict(track) for lane, track in driver.tracks.items()}})
        step(queue.popleft())
        if screen is not None:
            render(screen, state)
    result = {"seed": seed, "policy": policy, "distance": round(state.distance, 3),
              "ticks": state.ticks, "crashed": state.crashed,
              "final_speed": round(state.ego.velocity.x, 3), "requests": len(latencies),
              "mean_decision_ms": round(statistics.mean(latencies), 3),
              "max_decision_ms": round(max(latencies), 3)}
    if trace:
        result["trace"] = history
    return result


def summarize(results):
    distances = [r["distance"] for r in results]
    return {"games": len(results), "survived": sum(not r["crashed"] and r["ticks"] == 3600 for r in results),
            "mean_distance": round(statistics.mean(distances), 3),
            "median_distance": round(statistics.median(distances), 3),
            "min_distance": min(distances), "max_distance": max(distances),
            "mean_decision_ms": round(statistics.mean(r["mean_decision_ms"] for r in results), 3)}


def provenance():
    return {"controller_sha256": hashlib.sha256(Path(__file__).with_name("expert.py").read_bytes()).hexdigest(),
            "python": sys.version.split()[0],
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pygame", "fastapi", "pydantic")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", help="Explicit string seeds")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--prefix", default="tune-")
    parser.add_argument("--policy", choices=("expert", "accelerate", "coast"), default="expert")
    parser.add_argument("--config", type=Path, help="Load parameter JSON; explicit flags override it")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--margin", type=float)
    parser.add_argument("--uncertainty", type=float)
    parser.add_argument("--blind-history", type=int)
    parser.add_argument("--max-ticks", type=int, default=3600)
    parser.add_argument("--sensor-removal", type=int, default=0)
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.games < 1 or not 1 <= args.max_ticks <= 3600 or not 0 <= args.sensor_removal <= 16:
        parser.error("Require games >= 1, 1 <= max-ticks <= 3600, and 0 <= sensor-removal <= 16")
    if not args.visual:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    parameters = asdict(load_config(args.config) if args.config else Config())
    for name in ("horizon", "batch_size", "margin", "uncertainty", "blind_history"):
        if getattr(args, name) is not None:
            parameters[name] = getattr(args, name)
    config = Config(**parameters)
    results = []
    try:
        for seed in args.seeds or [f"{args.prefix}{i}" for i in range(args.games)]:
            result = run_episode(seed, args.policy, config, args.max_ticks, args.visual, args.trace, args.sensor_removal)
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "trace"}), flush=True)
    finally:
        pygame.quit()
    report = {"provenance": provenance(), "config": asdict(config), "sensor_removal": args.sensor_removal,
              "max_ticks": args.max_ticks, "summary": summarize(results), "results": results}
    print(json.dumps(report["summary"], indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()