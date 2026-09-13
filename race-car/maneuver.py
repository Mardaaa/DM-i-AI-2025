"""Bounded two-stage motion primitives and batch-end recovery screening.

No simulator imports, stochastic decisions, or hidden state. Keep the legacy
planner selectable: expanding the action space is not automatically safer.
"""

import numpy as np

from expert import ACTIONS, LANE_CENTERS


def rollouts(y, vx, vy, targets, throttles, delays, prefix_actions, horizon):
    """Exact action-before-motion integration, including speed clipping per tick.

    A prefix can brake/accelerate/coast or directly counter-steer; afterwards
    bang-bang lane settling interleaves the selected throttle whenever settled.
    Arrays hold one path per row. Pure function, also used for recovery paths.
    """
    size = len(targets)
    targets = np.asarray(targets)
    throttles = np.asarray(throttles)
    delays = np.asarray(delays)
    prefix_actions = np.asarray(prefix_actions)
    py, sx, sy = (np.full(size, float(value)) for value in (y, vx, vy))
    actions = np.empty((size, horizon), dtype=np.int8)
    ys, speeds, vys, distances = (np.empty((size, horizon)) for _ in range(4))
    travelled = np.zeros(size)
    for t in range(horizon):
        stop = py + sy * np.abs(sy) / 0.2
        lateral = np.where(targets > stop, 4, 3)
        settled = (np.abs(targets - py) < 4) & (np.abs(sy) < 0.05)
        act = np.where(t < delays, prefix_actions, np.where(settled, throttles, lateral))
        sx = np.maximum(0, sx + 0.1 * ((act == 1).astype(float) - (act == 2)))
        sy += 0.1 * ((act == 4).astype(float) - (act == 3))
        py += sy
        travelled += sx
        actions[:, t], ys[:, t], speeds[:, t], vys[:, t], distances[:, t] = act, py, sx, sy, travelled
    return actions, ys, speeds, vys, distances


def primitives(delay):
    """Legacy lane/throttle paths plus throttle-first and counter-steer paths."""
    specs = []
    for target in LANE_CENTERS:
        for throttle in (1, 0, 2):
            specs.append((target, throttle, 0, 0))
            for duration in (delay, delay * 2):
                for prefix in (1, 0, 2):
                    specs.append((target, throttle, duration, prefix))
        # Short direct steering pulses let the planner reverse lateral momentum
        # before braking/accelerating, instead of committing to a lane center.
        for prefix in (3, 4):
            specs.append((target, 2, min(delay, 8), prefix))
    return tuple(np.asarray(column) for column in zip(*specs))


def safe_ticks(mask):
    return np.where(mask.any(axis=1), mask.argmax(axis=1), mask.shape[1])


def homogeneous_count(actions, maximum):
    count = min(maximum, len(actions))
    changes = np.flatnonzero(actions[:count] != actions[0])
    return int(changes[0]) if changes.size else count


def recoverable(driver, action, count, vx, vy, frontiers, remaining):
    """Does this batch leave at least one predicted collision-free continuation?

    Test all fifteen settle/throttle continuations. Prefix collision checks
    and absolute time/distance offsets are retained. This finite library is
    not a proof of viability under unobserved future traffic.
    """
    length = min(remaining, count + driver.config.recovery_ticks)
    targets = np.repeat(LANE_CENTERS, 3)
    throttles = np.tile((1, 0, 2), 5)
    arrays = rollouts(driver.y, vx, vy, targets, throttles, np.full(15, count),
                      np.full(15, action), length)
    if driver.config.planner == "drift":
        from drift import extend
        arrays, targets = extend(arrays, targets, driver, vx, vy, length, action, count)
    _, ys, speeds, _, distances = arrays
    mask = driver.collision_mask(ys, speeds, distances, targets, vx, frontiers)
    return bool((~mask.any(axis=1)).any()), int(safe_ticks(mask).max())


def choose_actions(driver, vx, vy):
    remaining = max(1, 3600 - driver.tick)
    horizon = min(driver.config.horizon, remaining)
    targets, throttles, delays, prefix_actions = primitives(driver.config.maneuver_delay)
    arrays = rollouts(driver.y, vx, vy, targets, throttles, delays, prefix_actions, horizon)
    original_count = len(targets)
    if driver.config.planner == "drift":
        from drift import extend
        arrays, targets = extend(arrays, targets, driver, vx, vy, horizon)
        extra_count = len(targets) - original_count
        delays = np.concatenate((delays, np.zeros(extra_count, dtype=int)))
        prefix_actions = np.concatenate((prefix_actions, np.zeros(extra_count, dtype=int)))
    actions, ys, speeds, _, distances = arrays
    frontiers = driver._unseen_frontiers()
    mask = driver.collision_mask(ys, speeds, distances, targets, vx, frontiers)
    safe = safe_ticks(mask)
    terminal_weight = min(driver.config.terminal_speed_weight, max(0, remaining - horizon))
    value = distances[:, -1] + terminal_weight * speeds[:, -1]
    value -= driver.config.lane_change_penalty * np.abs(targets - driver.y)
    # Lexicographic ranking avoids finite penalty weights losing safety priority.
    ranked = np.lexsort((-np.arange(len(value)), value, safe))[::-1]
    recoveries = {}
    options = []
    for index in ranked:
        count = homogeneous_count(actions[index], min(driver.config.batch_size, remaining))
        if driver.config.adaptive_batch and safe[index] < driver.config.batch_size * 4:
            count = min(count, max(1, int(safe[index]) // 4))
        key = (int(actions[index, 0]), count)
        if key not in recoveries:
            recoveries[key] = recoverable(driver, *key, vx, vy, frontiers, remaining)
        recovery, recovery_safe = recoveries[key]
        options.append((int(index), count, recovery, recovery_safe))
        if safe[index] == horizon and recovery:
            break
    # If no fully safe candidate exists, retain the best available safe duration;
    # do not turn a heuristic screening failure into an empty/invalid response.
    eligible = [option for option in options if option[2]]
    selected = eligible[0] if eligible else max(options, key=lambda o: (min(safe[o[0]], o[3]), value[o[0]]))
    index, count, recovery, recovery_safe = selected
    result = [ACTIONS[int(actions[index, 0])]] * count
    diagnostics = {"planner": driver.config.planner, "target": float(targets[index]), "safe_ticks": int(safe[index]),
                   "recovery_safe": recovery, "recovery_safe_ticks": recovery_safe,
                   "prefix_action": ACTIONS[int(prefix_actions[index])], "prefix_ticks": int(delays[index]),
                   "candidate_count": len(targets), "recovery_checks": len(recoveries),
                   "recovery_vetoes": sum(not feasible for feasible, _ in recoveries.values()),
                   "motion_family": "drift" if index >= original_count else "settle",
                   "horizon": horizon, "terminal_weight": terminal_weight,
                   "y": driver.y, "tracks": len(driver.tracks)}
    return result, diagnostics