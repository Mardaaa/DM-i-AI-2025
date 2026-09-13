from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from expert import Config, ExpertController, LANE_CENTERS, load_config
from maneuver import choose_actions, homogeneous_count, primitives, recoverable, rollouts


def driver():
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "expert.json")
    instance = ExpertController(replace(config, planner="maneuver"))
    instance.tick = 0
    instance.speed_history = [(0, 10.0)]
    # For constructed motion tests there is no sensed evidence about traffic.
    instance.sensors = {}
    return instance


def test_candidate_library_contains_legacy_and_brake_then_shift():
    targets, throttles, delays, prefixes = primitives(24)
    assert len(targets) == 115
    for target in LANE_CENTERS:
        for throttle in (0, 1, 2):
            assert ((targets == target) & (throttles == throttle) & (delays == 0)).any()
    assert ((targets == 375) & (delays == 24) & (prefixes == 2)).any()


def test_mixed_throttle_clips_each_tick_before_accelerating():
    actions, ys, speeds, vys, distances = rollouts(599, 0.1, 0, [599], [1], [5], [2], 10)
    assert actions[0].tolist() == [2] * 5 + [1] * 5
    assert speeds[0].tolist() == pytest.approx([0] * 5 + [0.1, 0.2, 0.3, 0.4, 0.5])
    assert distances[0, -1] == pytest.approx(1.5)
    assert np.all(ys == 599) and np.all(vys == 0)


def test_rollouts_match_legacy_and_preserve_momentum():
    d = driver()
    old_actions, old_y, old_speeds, old_distance, targets = d._plans(10, 0)
    actions, ys, speeds, _, distance = rollouts(d.y, 10, 0, targets, np.tile((1, 0, 2), 5),
                                              np.zeros(15), np.zeros(15), d.config.horizon)
    np.testing.assert_array_equal(actions, old_actions)
    np.testing.assert_allclose(ys, old_y)
    # Legacy cumsum and per-tick clipping differ by roundoff near zero.
    np.testing.assert_allclose(speeds, old_speeds, atol=1e-12)
    np.testing.assert_allclose(distance, old_distance)
    _, y, _, vy, _ = rollouts(599, 10, 2, [375], [1], [3], [2], 3)
    assert y[0].tolist() == [601, 603, 605]
    assert vy[0].tolist() == [2, 2, 2]


def test_recovery_keeps_prefix_and_absolute_time(monkeypatch):
    d = driver()
    captured = {}

    def mask(ys, speeds, distances, targets, vx, frontiers):
        captured.update(ys=ys, speeds=speeds, distances=distances)
        return np.zeros(ys.shape, dtype=bool)

    monkeypatch.setattr(d, "collision_mask", mask)
    feasible, ticks = recoverable(d, 1, 8, 10, 2, {}, 3600)
    assert feasible and ticks == 8 + d.config.recovery_ticks
    assert np.allclose(captured["ys"][:, 7], 615)
    assert np.allclose(captured["speeds"][:, 7], 10.8)
    assert np.allclose(captured["distances"][:, 7], 83.6)


def test_recovery_rejects_prefix_collision(monkeypatch):
    d = driver()

    def mask(ys, *args):
        blocked = np.zeros(ys.shape, dtype=bool)
        blocked[:, 2] = True
        return blocked

    monkeypatch.setattr(d, "collision_mask", mask)
    assert recoverable(d, 1, 8, 10, 0, {}, 3600) == (False, 2)


def test_recovery_can_veto_high_scoring_prefix(monkeypatch):
    import maneuver
    d = driver()
    monkeypatch.setattr(d, "collision_mask", lambda ys, *args: np.zeros(ys.shape, dtype=bool))
    monkeypatch.setattr(maneuver, "recoverable", lambda instance, action, count, *args: (action != 1, 120))
    actions, info = choose_actions(d, 10, 0)
    assert actions[0] != "ACCELERATE"
    assert info["recovery_safe"]


def test_last_tick_has_no_terminal_reward(monkeypatch):
    d = driver()
    d.tick = 3599
    monkeypatch.setattr(d, "collision_mask", lambda ys, *args: np.zeros(ys.shape, dtype=bool))
    actions, info = choose_actions(d, 10, 0)
    assert actions == ["ACCELERATE"]
    assert info["horizon"] == 1 and info["terminal_weight"] == 0
    assert info["recovery_safe_ticks"] == 1


def test_homogeneous_batch_ends_before_action_change():
    assert homogeneous_count(np.array([1, 1, 3, 3]), 8) == 2
    assert homogeneous_count(np.array([1, 1, 1]), 2) == 2


@pytest.mark.parametrize("kwargs", [{"planner": "unknown"}, {"maneuver_delay": 0}, {"recovery_ticks": 301}])
def test_invalid_maneuver_config(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_delayed_lane_change_cannot_skip_current_lane_reserve():
    d = driver()
    d.tick = 140
    d.speed_history = [(0, 10), (140, 30)]
    targets = np.array([375])
    arrays = rollouts(599, 30, 0, targets, [1], [195], [1], 195)
    _, ys, speeds, _, distances = arrays
    mask = d.collision_mask(ys, speeds, distances, targets, 30, {2: (1180, -1180)})
    assert mask.any()
    # Legacy compares only destination lane: this is the loophole being closed.
    d.config = replace(d.config, planner="legacy")
    assert not d.collision_mask(ys, speeds, distances, targets, 30, {2: (1180, -1180)}).any()


def test_maneuver_matches_original_mixed_action_physics(monkeypatch):
    import pygame
    from benchmark import step
    from src.game import core
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    try:
        core.initialize_game_state("", "maneuver-motion-test")
        core.STATE.ego.velocity.x = 0.1
        arrays = rollouts(599, 0.1, 0, [375], [1], [12], [2], 160)
        actions, ys, speeds, vys, distances = arrays
        from expert import ACTIONS
        for t in range(160):
            state = step(ACTIONS[int(actions[0, t])])
            assert state.ego.y + 89 == pytest.approx(ys[0, t])
            assert state.ego.velocity.x == pytest.approx(speeds[0, t])
            assert state.ego.velocity.y == pytest.approx(vys[0, t])
            assert state.distance == pytest.approx(distances[0, t])
    finally:
        pygame.quit()


def test_maneuver_replay_and_api(monkeypatch):
    import pygame
    from fastapi.testclient import TestClient
    import api
    from benchmark import observation, run_episode
    from src.game import core
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    config = replace(driver().config, maneuver_delay=12)
    pygame.init()
    try:
        first = run_episode("maneuver-replay", config=config, max_ticks=240, trace=True)
        second = run_episode("maneuver-replay", config=config, max_ticks=240, trace=True)
        assert first["trace"] == second["trace"]
        assert first["distance"] == second["distance"]
        assert all(len(set(frame["actions"])) == 1 for frame in first["trace"])
        core.initialize_game_state("", "maneuver-api")
        obs = observation(core.STATE)
        monkeypatch.setattr(api, "controller", ExpertController(config))
        expected = ExpertController(config).actions(obs)
        client = TestClient(api.app)
        response = client.post("/predict", json=obs)
        assert response.status_code == 200
        assert response.json() == {"actions": expected}
        assert client.post("/predict", json=obs).json() == response.json()
    finally:
        pygame.quit()