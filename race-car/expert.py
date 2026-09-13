"""Sensor-only, deterministic model-predictive driver. No training or RNG access.

Coordinates follow the game: x points forward, y points towards STEER_RIGHT.
Distances are pixels, velocities pixels/tick, acceleration 0.1 pixels/tick².
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


ANGLES = {
    "left_side": 0, "left_side_front": 22.5, "left_front": 45,
    "front_left_front": 67.5, "front": 90, "front_right_front": 112.5,
    "right_front": 135, "right_side_front": 157.5, "right_side": 180,
    "right_side_back": 202.5, "right_back": 225, "back_right_back": 247.5,
    "back": 270, "back_left_back": 292.5, "left_back": 315,
    "left_side_back": 337.5,
}
RAYS = {name: (math.sin(math.radians(a)), -math.cos(math.radians(a)))
        for name, a in ANGLES.items()}
LANE_CENTERS = (151.0, 375.0, 599.0, 823.0, 1047.0)
CAR_WIDTH = 360
CAR_HEIGHT = 179
ACTIONS = ("NOTHING", "ACCELERATE", "DECELERATE", "STEER_LEFT", "STEER_RIGHT")


@dataclass
class Config:
    horizon: int = 210
    batch_size: int = 12
    margin: float = 12.0
    uncertainty: float = 0.06
    terminal_speed_weight: float = 80.0
    blind_history: int = 150
    blind_speed_buffer: float = 5.5
    stale_velocity_uncertainty: float = 0.12
    lane_change_penalty: float = 0.1
    planner: str = "legacy"
    maneuver_delay: int = 24
    recovery_ticks: int = 120
    adaptive_batch: bool = True
    drift_speed: float = 2.0

    def __post_init__(self):
        if self.planner not in ("legacy", "maneuver", "drift"):
            raise ValueError("planner must be legacy, maneuver or drift")
        if not math.isfinite(self.drift_speed) or not 0.2 <= self.drift_speed <= 6:
            raise ValueError("drift_speed must be finite and between 0.2 and 6")
        if not 1 <= self.maneuver_delay <= 120 or not 1 <= self.recovery_ticks <= 300:
            raise ValueError("maneuver_delay must be 1..120; recovery_ticks must be 1..300")
        if not 1 <= self.batch_size <= self.horizon <= 1000:
            raise ValueError("Require 1 <= batch_size <= horizon <= 1000")
        if not 1 <= self.blind_history <= 250:
            raise ValueError("blind_history must be between 1 and 250 ticks")
        if any(not math.isfinite(v) or v < 0 for v in
               (self.margin, self.uncertainty, self.terminal_speed_weight,
                self.blind_speed_buffer, self.stale_velocity_uncertainty, self.lane_change_penalty)):
            raise ValueError("Safety margins and scoring weights must be finite and nonnegative")


def load_config(path: str | Path) -> Config:
    """Load an explicit parameter JSON or a benchmark's config section."""
    data = json.loads(Path(path).read_text())
    return Config(**data.get("config", data))


@dataclass
class Track:
    x: float
    speed: float
    spread: float
    seen: int
    anchor_x: float | None = None
    anchor_tick: int = 0
    anchor_distance: float = 0.0


def lateral_action(y: float, vy: float, target: float) -> int:
    """Bang-bang steering, with explicit braking of lateral momentum."""
    # Sensor origins are integer rect centers. A small deadband prevents
    # quantization chatter from starving ACCELERATE/DECELERATE indefinitely.
    if abs(target - y) < 4.0 and abs(vy) < 0.05:
        return 0
    stop = y + vy * abs(vy) / 0.2
    return 4 if target > stop else 3


class ExpertController:
    """One instance per game stream. Only the public request DTO is consumed."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.reset()

    def reset(self):
        self.tick = -1
        self.distance = 0.0
        self.y = 599.0
        self.tracks: dict[int, Track] = {}
        self.previous_actions: list[str] = []
        self.previous_vy = 0.0
        self.last_plan = {}
        self.speed_history: list[tuple[int, float]] = []
        self.sensors = {}
        self.last_request = None

    def _position(self, sensors, dt: int) -> float:
        # Integrate our own actions first. Rect centers truncate the floating y.
        predicted = self.y
        vy = self.previous_vy
        for i in range(dt):
            action = self.previous_actions[min(i, len(self.previous_actions) - 1)] if self.previous_actions else "NOTHING"
            vy += 0.1 * ((action == "STEER_RIGHT") - (action == "STEER_LEFT"))
            predicted += vy
        lower, upper = 129.0, 1070.0
        for name, reading in sensors.items():
            if reading is None or name not in RAYS:
                continue
            _, dy = RAYS[name]
            if dy < -1e-6:
                lower = max(lower, 40 - dy * reading)
            elif dy > 1e-6:
                upper = min(upper, 1160 - dy * reading)
        # Rays hitting cars give bounds; wall hits make the bounds coincide.
        if upper - lower < 1.1:
            return (lower + upper) / 2
        return min(upper, max(lower, predicted))

    def _observe(self, state):
        tick = int(state["elapsed_ticks"])
        if tick <= self.tick or float(state["distance"]) < self.distance:
            self.reset()
        dt = max(0, tick - self.tick) if self.tick >= 0 else 0
        distance = float(state["distance"])
        travelled = distance - self.distance
        sensors = state["sensors"]
        self.sensors = sensors
        self.y = self._position(sensors, dt)
        for track in self.tracks.values():
            track.x += track.speed * dt - travelled
            track.spread += 0.15 * dt

        measurements: dict[int, tuple[float, float]] = {}
        for name, reading in sensors.items():
            if reading is None or name not in RAYS:
                continue
            dx, dy = RAYS[name]
            px, py = dx * reading, self.y + dy * reading
            if abs(py - 40) < 2 or abs(py - 1160) < 2:
                continue
            for lane, center in enumerate(LANE_CENTERS):
                top, bottom = center - 89, center + 90
                if top - 1 <= py <= bottom + 1:
                    if min(abs(py - top), abs(py - bottom)) > 1.5 and abs(dx) > 1e-6:
                        x = px + math.copysign(180, dx)
                        lo, hi = x - 1, x + 1
                    else:
                        lo, hi = px - 180, px + 180
                    old_lo, old_hi = measurements.get(lane, (-math.inf, math.inf))
                    measurements[lane] = max(lo, old_lo), min(hi, old_hi)
                    break

        vx = float(state["velocity"]["x"])
        self.speed_history.append((tick, vx))
        self.speed_history = [(t, v) for t, v in self.speed_history if t >= tick - 250]
        for lane, (lo, hi) in measurements.items():
            if lo > hi + 3:
                continue
            track = self.tracks.get(lane)
            if track is None or lo > track.x + track.spread + 250 or hi < track.x - track.spread - 250:
                x = (lo + hi) / 2
                initial_speed = min(v for _, v in self.speed_history) - 2.5 if x > 0 else vx + 2.5
                track = Track(x, initial_speed, (hi - lo) / 2, tick, anchor_tick=tick)
                self.tracks[lane] = track
            track.x = min(hi, max(lo, track.x))
            track.spread = min(track.spread, max(1, (hi - lo) / 2))
            track.seen = tick
            if hi - lo <= 3:
                x = (lo + hi) / 2
                if track.anchor_x is not None and tick > track.anchor_tick:
                    elapsed = tick - track.anchor_tick
                    measured_speed = (x - track.anchor_x + distance - track.anchor_distance) / elapsed
                    track.speed = measured_speed
                track.x = x
                track.anchor_x, track.anchor_tick, track.anchor_distance = x, tick, distance
        self.tracks = {lane: track for lane, track in self.tracks.items()
                       if -2000 < track.x < 2300 and tick - track.seen < 500}
        self.tick, self.distance = tick, distance

    def _unseen_frontiers(self):
        """Inverse ray casting: nearest car that could hide between/behind rays.

        Used for blind lane changes and the current-lane braking reserve.
        A null reading is clear to 1000px; an absent key gives no evidence.
        """
        grid = np.arange(-1800.0, 2001.0, 20.0)
        frontiers = {}
        for lane, center in enumerate(LANE_CENTERS):
            if lane in self.tracks:
                continue
            possible = np.ones(grid.size, dtype=bool)
            for name, reading in self.sensors.items():
                if name not in RAYS:
                    continue
                dx, dy = RAYS[name]
                if abs(dy) < 1e-8:
                    if not center - 89 <= self.y <= center + 90:
                        continue
                    near_y, far_y = 0.0, 1000.0
                else:
                    ends = ((center - 89 - self.y) / dy, (center + 90 - self.y) / dy)
                    near_y, far_y = max(0.0, min(ends)), min(1000.0, max(ends))
                if far_y < near_y:
                    continue
                if abs(dx) < 1e-8:
                    hit = np.where(np.abs(grid) <= 180, near_y, math.inf)
                else:
                    a, b = (grid - 180) / dx, (grid + 180) / dx
                    near = np.maximum(np.minimum(a, b), near_y)
                    far = np.minimum(np.maximum(a, b), far_y)
                    hit = np.where(near <= far, near, math.inf)
                possible &= hit >= (1000.0 if reading is None else reading) - 2
            front = grid[possible & (grid >= 0)]
            back = grid[possible & (grid < 0)]
            frontiers[lane] = (float(front.min()) if front.size else 2200.0,
                               float(back.max()) if back.size else -2200.0)
        return frontiers

    def _plans(self, vx: float, vy: float):
        """Enumerate fixed lane targets and accelerate/coast/brake policies."""
        horizon = self.config.horizon
        lateral_paths, action_paths, targets = [], [], []
        for target in LANE_CENTERS:
            y, speed_y = self.y, vy
            ys, actions = [], []
            for _ in range(horizon):
                action = lateral_action(y, speed_y, target)
                speed_y += 0.1 * ((action == 4) - (action == 3))
                y += speed_y
                ys.append(y)
                actions.append(action)
            for throttle in (1, 0, 2):
                lateral_paths.append(ys)
                action_paths.append([a if a else throttle for a in actions])
                targets.append(target)
        actions = np.asarray(action_paths, dtype=np.int8)
        ys = np.asarray(lateral_paths)
        acceleration = 0.1 * ((actions == 1).astype(float) - (actions == 2))
        speeds = vx + np.cumsum(acceleration, axis=1)
        # All throttle policies have one sign, so clipping is exact at zero.
        speeds = np.maximum(speeds, 0)
        distances = np.cumsum(speeds, axis=1)
        return actions, ys, speeds, distances, np.asarray(targets)

    def collision_mask(self, ys, speeds, distances, targets, vx, frontiers=None):
        """Predicted collisions from the current observation, for arbitrary paths.

        All distances and times start at the observation, including recovery
        branches. Uses estimated traffic only; uncertainty is heuristic.
        """
        horizon = ys.shape[1]
        times = np.arange(1, horizon + 1)
        collision = (ys < 131) | (ys > 1067)
        for lane, track in self.tracks.items():
            xs = track.x + track.speed * times - distances
            age = max(0, self.tick - track.anchor_tick)
            padding = (self.config.margin + track.spread + self.config.uncertainty * times ** 1.5
                       + self.config.stale_velocity_uncertainty * math.sqrt(age) * times)
            collision |= (np.abs(xs) < CAR_WIDTH + padding) & (np.abs(ys - LANE_CENTERS[lane]) < CAR_HEIGHT + 3)
        # Unknown does not mean empty: sparse rays have substantial blind spots.
        for lane, (front, back) in (self._unseen_frontiers() if frontiers is None else frontiers).items():
            if abs(self.y - LANE_CENTERS[lane]) < CAR_HEIGHT + 3:
                # Reserve braking range for a lead car just outside visibility.
                floor_speed = (min(v for t, v in self.speed_history if t >= self.tick - self.config.blind_history)
                               - self.config.blind_speed_buffer)
                reaction = min(self.config.batch_size, horizon)
                future = np.maximum(0, times - reaction)
                brake_distance = (distances[:, reaction - 1, None]
                                  + speeds[:, reaction - 1, None] * future
                                  - 0.05 * future * (future + 1))
                closing = np.maximum(0, speeds[:, reaction - 1] - floor_speed)
                stopping = closing / 0.1
                brake_distance = np.where(times <= reaction, distances, brake_distance)
                phantom_x = front + floor_speed * times - brake_distance
                danger = (phantom_x < CAR_WIDTH + self.config.margin) & (times <= reaction + stopping[:, None])
                staying = targets == LANE_CENTERS[lane]
                if self.config.planner != "legacy":
                    # A future destination does not exempt a throttle-first
                    # prefix from preserving braking range in its present lane.
                    collision |= danger & (np.abs(ys - LANE_CENTERS[lane]) < CAR_HEIGHT + 3)
                else:
                    collision |= danger & staying[:, None]
                continue
            overlap = np.abs(ys - LANE_CENTERS[lane]) < CAR_HEIGHT + 3
            forward = front + (vx - 8) * times - distances
            rearward = back + (vx + 5) * times - distances
            collision |= overlap & ((np.abs(forward) < CAR_WIDTH + 20) | (np.abs(rearward) < CAR_WIDTH + 20))
        return collision

    def actions(self, state: dict) -> list[str]:
        if state.get("did_crash", False):
            self.reset()
            return ["NOTHING"]
        key = (state["elapsed_ticks"], state["distance"], state["velocity"]["x"],
               state["velocity"]["y"], tuple(sorted(state["sensors"].items())))
        if key == self.last_request:
            return list(self.previous_actions)
        self._observe(state)
        vx, vy = float(state["velocity"]["x"]), float(state["velocity"]["y"])
        if self.config.planner != "legacy":
            from maneuver import choose_actions
            result, self.last_plan = choose_actions(self, vx, vy)
            self.previous_actions, self.previous_vy = result, vy
            self.last_request = key
            return result
        actions, ys, speeds, distances, targets = self._plans(vx, vy)
        collision = self.collision_mask(ys, speeds, distances, targets, vx)
        first_hit = np.where(collision.any(axis=1), collision.argmax(axis=1), self.config.horizon)
        value = distances[:, -1] + self.config.terminal_speed_weight * speeds[:, -1]
        value -= self.config.lane_change_penalty * np.abs(targets - self.y)
        # Safety dominates distance. If trapped, buy the most reaction time.
        value += first_hit * 100000.0
        best = int(np.argmax(value))
        count = min(self.config.batch_size, max(1, 3600 - self.tick))
        # Homogeneous batches are safe under both FIFO and the starter's LIFO
        # playback. Never ship a turn followed by its counter-turn in one batch.
        changes = np.flatnonzero(actions[best, :count] != actions[best, 0])
        if changes.size:
            count = int(changes[0])
        result = [ACTIONS[int(a)] for a in actions[best, :count]]
        self.previous_actions, self.previous_vy = result, vy
        self.last_request = key
        self.last_plan = {"target": float(targets[best]), "safe_ticks": int(first_hit[best]),
                          "y": self.y, "tracks": len(self.tracks)}
        return result