import os
from pathlib import Path
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pygame
import pytest
from fastapi.testclient import TestClient

from expert import ACTIONS, Config, ExpertController, lateral_action
from benchmark import observation, run_episode, step
from src.game import core
from src.mathematics.collision import Line, get_intersection_point, intersects
from src.mathematics.vector import Vector


@pytest.fixture(autouse=True)
def game():
    pygame.init()
    core.initialize_game_state("", "unit-test")
    yield core.STATE
    pygame.quit()


def test_geometry_matches_assets_and_wall_sensors(game):
    assert game.ego.rect.size == (360, 179)
    assert game.ego.rect.center == (800, 599)
    obs = observation(game)
    assert obs["sensors"]["left_side"] == pytest.approx(559)
    assert obs["sensors"]["right_side"] == pytest.approx(561)
    assert obs["sensors"]["front"] is None
    assert obs["sensors"]["back"] is None


def test_ray_intersection_and_touching_rectangles():
    ray = Line(Vector(0, 0), Vector(10, 0))
    edge = Line(Vector(5, -1), Vector(5, 1))
    point = get_intersection_point(ray, edge)
    assert point.to_array() == [5, 0]
    assert get_intersection_point(ray, Line(Vector(0, 1), Vector(10, 1))) is None
    assert not intersects(pygame.Rect(0, 0, 360, 179), pygame.Rect(360, 0, 360, 179))
    assert intersects(pygame.Rect(0, 0, 360, 179), pygame.Rect(359, 0, 360, 179))


def test_observer_reconstructs_traffic_from_only_dto(game):
    car = game.car_bucket.pop()
    car.x, car.y, car.lane = 1420, 510, game.road.lanes[2]
    game.cars.append(car)
    obs = observation(game)
    driver = ExpertController()
    driver.actions(obs)
    assert obs["sensors"]["front"] == pytest.approx(620)
    assert driver.y == pytest.approx(599)
    assert driver.tracks[2].x == pytest.approx(800)
    assert driver.tracks[2].spread <= 1


@pytest.mark.parametrize("target", [151, 375, 599, 823, 1047])
def test_lateral_controller_settles_without_chatter(target):
    y, vy = 599.0, 0.0
    for _ in range(250):
        action = lateral_action(y, vy, target)
        vy += 0.1 * ((action == 4) - (action == 3))
        y += vy
        assert 129 <= y <= 1070
    assert abs(y - target) < 4
    assert vy == pytest.approx(0, abs=1e-10)
    assert lateral_action(y, vy, target) == 0


def test_planned_motion_matches_original_physics(game):
    driver = ExpertController()
    actions, ys, speeds, distances, targets = driver._plans(10, 0)
    plan = int(np.flatnonzero(targets == 375)[0])
    for i in range(100):
        step(ACTIONS[int(actions[plan, i])])
        assert game.ego.y + 89 == pytest.approx(ys[plan, i])
        assert game.ego.velocity.x == pytest.approx(speeds[plan, i])
        assert game.distance == pytest.approx(distances[plan, i])


def test_headless_step_preserves_original_update_and_rng(game):
    actions = ["ACCELERATE"] * 30 + ["STEER_LEFT"] * 10 + ["STEER_RIGHT"] * 10

    def snapshot():
        state = core.STATE
        return (state.distance, state.ego.y,
                [(c.x, c.y, c.velocity.x, c.velocity.y) for c in state.cars])

    for action in actions:
        core.update_game(action)
    expected = snapshot()
    core.initialize_game_state("", "unit-test")
    for action in actions:
        step(action)
    assert snapshot() == expected


def test_collision_checked_every_tick(game):
    car = game.car_bucket.pop()
    car.x, car.y, car.lane = game.ego.x, game.ego.y, game.road.lanes[2]
    game.cars.append(car)
    assert step("NOTHING").crashed


def test_api_schema_retry_and_input_validation(game):
    import api
    from settings import driver_config

    api.controller.reset()
    assert api.controller.config == driver_config()
    client = TestClient(api.app)
    obs = observation(game)
    response = client.post("/predict", json=obs)
    assert response.status_code == 200
    assert set(response.json()) == {"actions"}
    assert 1 <= len(response.json()["actions"]) <= Config.batch_size
    assert len(set(response.json()["actions"])) == 1
    assert client.post("/predict", json=obs).json() == response.json()
    assert client.get("/").status_code == 200
    assert client.get("/api").status_code == 200
    invalid = {**obs, "velocity": {"x": 10}}
    assert client.post("/predict", json=invalid).status_code == 422
    assert client.post("/predict", json={**obs, "distance": -1}).status_code == 422


def test_retry_preserves_tracks_and_new_game_resets(game):
    driver = ExpertController()
    obs = observation(game)
    first = driver.actions(obs)
    before = (driver.tick, driver.y, dict(driver.tracks))
    assert driver.actions(obs) == first
    assert (driver.tick, driver.y, driver.tracks) == before
    for action in first:
        step(action)
    driver.actions(observation(game))
    assert driver.tick > 0
    assert driver.actions(obs) == first
    assert driver.tick == 0
    assert driver.actions({**obs, "did_crash": True}) == ["NOTHING"]
    assert driver.tick == -1


def test_repeatable_seed_and_batches():
    a = run_episode("replay-test", max_ticks=600, trace=True)
    b = run_episode("replay-test", max_ticks=600, trace=True)
    for key in ("ticks", "distance", "crashed", "final_speed", "trace"):
        assert a[key] == b[key]
    for frame in a["trace"]:
        actions = frame["actions"]
        assert actions == list(reversed(actions))
        assert set(actions) <= set(ACTIONS)


@pytest.mark.parametrize("seed", ["tune-9", "validation-45"])
def test_full_game_regressions(seed):
    result = run_episode(seed)
    assert not result["crashed"]
    assert result["ticks"] == 3600


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"horizon": 2, "batch_size": 3},
                                    {"blind_history": 0}, {"margin": float("nan")}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)