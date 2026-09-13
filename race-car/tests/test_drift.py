from dataclasses import replace

import numpy as np
import pytest

from drift import primitives, rollouts
from expert import ACTIONS, Config, ExpertController
from maneuver import choose_actions, recoverable, rollouts as settling_rollouts


@pytest.mark.parametrize("speed", [1, 2, 3])
def test_drift_accelerates_during_lane_change_and_settles(speed):
    actions, ys, _, vys, distances = rollouts(599, 10, 0, [375], [1], [speed], 400)
    old = settling_rollouts(599, 10, 0, [375], [1], [0], [0], 400)
    assert ((actions == 1) & (np.abs(vys) > 0.05)).any()
    assert abs(ys[0, -1] - 375) < 4
    assert abs(vys[0, -1]) < 0.05
    assert np.isin(actions, [3, 4]).sum() < np.isin(old[0], [3, 4]).sum()
    assert distances[0, -1] > old[-1][0, -1]


def test_library_includes_off_center_targets_within_road():
    targets, throttles, caps = primitives(2)
    assert len(targets) == len(throttles) == len(caps) == 135
    assert set(targets) >= {375, 351, 399}
    assert targets.min() >= 131 and targets.max() <= 1067
    assert set(caps) == {1, 2, 3}


def test_prefix_clipping_and_lateral_momentum():
    actions, y, vx, vy, distance = rollouts(599, 0.1, 2, [375], [1], [2], 8, 2, 8)
    assert np.all(actions == 2)
    np.testing.assert_allclose(vx, 0)
    np.testing.assert_allclose(vy, 2)
    assert y[0, -1] == 615 and distance[0, -1] == 0


@pytest.mark.parametrize("speed", [0, 0.1, 7, float("inf"), float("nan")])
def test_invalid_drift_speed(speed):
    with pytest.raises(ValueError):
        Config(planner="drift", drift_speed=speed)


def test_recovery_contains_drift_and_keeps_prefix(monkeypatch):
    driver = ExpertController(Config(planner="drift"))
    captures = []

    def mask(ys, speeds, distances, *args):
        assert ys.shape[0] == 150
        np.testing.assert_allclose(ys[:, 7], 615)
        np.testing.assert_allclose(speeds[:, 7], 10.8)
        np.testing.assert_allclose(distances[:, 7], 83.6)
        blocked = np.ones(ys.shape, dtype=bool)
        blocked[-1] = False  # Only a drift continuation is feasible.
        captures.append(True)
        return blocked

    monkeypatch.setattr(driver, "collision_mask", mask)
    assert recoverable(driver, 1, 8, 10, 2, {}, 3600)[0]
    assert captures


def test_drift_inherits_current_lane_blind_reserve():
    driver = ExpertController(Config(planner="drift"))
    driver.tick = 140
    driver.speed_history = [(0, 10), (140, 30)]
    _, ys, speeds, _, distances = rollouts(599, 30, 0, [375], [1], [2], 195, 1, 195)
    assert driver.collision_mask(ys, speeds, distances, np.array([375]), 30,
                                 {2: (1180, -1180)}).any()


def test_last_tick_and_candidate_count(monkeypatch):
    driver = ExpertController(Config(planner="drift"))
    driver.tick = 3599
    monkeypatch.setattr(driver, "_unseen_frontiers", lambda: {})
    monkeypatch.setattr(driver, "collision_mask", lambda ys, *args: np.zeros(ys.shape, dtype=bool))
    actions, info = choose_actions(driver, 10, 0)
    assert actions == ["ACCELERATE"]
    assert info["candidate_count"] == 250
    assert info["horizon"] == 1 and info["terminal_weight"] == 0


def test_drift_rollout_matches_original_physics(monkeypatch):
    import pygame
    from benchmark import step
    from src.game import core
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    try:
        core.initialize_game_state("", "drift-motion")
        actions, ys, speeds, vys, distances = rollouts(599, 10, 0, [375], [1], [2], 200)
        for t in range(200):
            state = step(ACTIONS[int(actions[0, t])])
            assert state.ego.y + 89 == pytest.approx(ys[0, t])
            assert state.ego.velocity.x == pytest.approx(speeds[0, t])
            assert state.ego.velocity.y == pytest.approx(vys[0, t])
            assert state.distance == pytest.approx(distances[0, t])
    finally:
        pygame.quit()


def test_drift_deterministic_replay_and_api(monkeypatch):
    import pygame
    from fastapi.testclient import TestClient
    import api
    from benchmark import observation, run_episode
    from src.game import core
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    config = replace(Config(), planner="drift", horizon=100, batch_size=8)
    pygame.init()
    try:
        first = run_episode("drift-replay", config=config, max_ticks=200, trace=True)
        second = run_episode("drift-replay", config=config, max_ticks=200, trace=True)
        assert first["trace"] == second["trace"]
        assert first["diagnostics"] == second["diagnostics"]
        assert all(len(set(frame["actions"])) == 1 for frame in first["trace"])
        core.initialize_game_state("", "drift-api")
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