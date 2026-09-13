"""Sensor-only pulse / throttle-drift / counter-steer motion primitives.

These paths supplement, never replace, the original maneuver library. All
paths are checked using the same traffic/wall model and recovery screening.
No simulator state, random draws, or learned model is used.
"""

import numpy as np

from expert import LANE_CENTERS


def primitives(speed):
    """135 paths: 15 off-center targets x 3 throttles x 3 cruise speeds.

    Offsets stay within a lane's free corridor when adjacent cars are abreast;
    they are not an assumption that a lane boundary is safe to straddle.
    """
    specs = [(np.clip(center + offset, 131, 1067), throttle,
              max(0.1, round(speed * factor, 1)))
             for center in LANE_CENTERS for offset in (-24, 0, 24)
             for throttle in (1, 0, 2) for factor in (0.5, 1, 1.5)]
    return tuple(np.asarray(column) for column in zip(*specs))


def rollouts(y, vx, vy, targets, throttles, cruise_speeds, horizon,
             prefix_action=0, prefix_ticks=0):
    """Exact per-tick physics, with optional constant-action recovery prefix.

    Pulse toward the target until reaching a cruise lateral speed, apply
    throttle while lateral momentum carries the car, then counter-steer near
    the stopping point. A 4px settling band avoids quantization chatter.
    """
    targets, throttles, cruise_speeds = map(np.asarray, (targets, throttles, cruise_speeds))
    size = len(targets)
    py, sx, sy = (np.full(size, float(value)) for value in (y, vx, vy))
    actions = np.empty((size, horizon), dtype=np.int8)
    ys, speeds, vys, distances = (np.empty((size, horizon)) for _ in range(4))
    travelled = np.zeros(size)
    for t in range(horizon):
        error = targets - py
        direction = np.where(error >= 0, 1, -1)
        toward = direction * sy
        # Moving away requires a reversal; approaching the stopping envelope
        # requires counter-steering. At cruise, acceleration costs no lateral
        # momentum and is therefore useful even far from a lane center.
        stopping = sy * sy / 0.2
        brake = (toward > 0.05) & (stopping >= np.maximum(0, np.abs(error) - 2))
        pulse = (toward < cruise_speeds - 0.05) | (toward < -0.05)
        lateral = np.where(brake, -direction, direction)
        act = np.where(brake | pulse, np.where(lateral > 0, 4, 3), throttles)
        settled = (np.abs(error) < 4) & (np.abs(sy) < 0.05)
        act = np.where(settled, throttles, act)
        if t < prefix_ticks:
            act = np.full(size, prefix_action, dtype=np.int8)
        sx = np.maximum(0, sx + 0.1 * ((act == 1).astype(float) - (act == 2)))
        sy += 0.1 * ((act == 4).astype(float) - (act == 3))
        py += sy
        travelled += sx
        actions[:, t], ys[:, t], speeds[:, t], vys[:, t], distances[:, t] = act, py, sx, sy, travelled
    return actions, ys, speeds, vys, distances


def extend(arrays, targets, driver, vx, vy, horizon, prefix_action=0, prefix_ticks=0):
    """Append drift paths for both main decisions and batch-end recovery."""
    drift_targets, throttles, cruises = primitives(driver.config.drift_speed)
    extra = rollouts(driver.y, vx, vy, drift_targets, throttles, cruises, horizon,
                     prefix_action, prefix_ticks)
    return (tuple(np.concatenate((old, new), axis=0) for old, new in zip(arrays, extra)),
            np.concatenate((targets, drift_targets)))