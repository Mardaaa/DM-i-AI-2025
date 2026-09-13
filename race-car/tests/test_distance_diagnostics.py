from collections import deque
import json

import numpy as np
import pygame
import pytest

import benchmark
from distance_diagnostics import DistanceDiagnostics
from expert import Config, ExpertController
from src.game import core
from src.mathematics import randomizer


@pytest.fixture
def game(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    yield
    pygame.quit()


def account(initial_speed, max_ticks, actions, speeds):
    diagnostics = DistanceDiagnostics(initial_speed, max_ticks)
    previous = initial_speed
    for action, speed in zip(actions, speeds):
        diagnostics.record_tick(action, previous, speed)
        previous = speed
    return diagnostics.result(sum(speeds))


def assert_identity(result):
    assert result["distance_loss"] == pytest.approx(
        result["action_opportunity_loss"] + result["termination_loss"], abs=1e-9)
    assert result["identity_residual"] == pytest.approx(0, abs=1e-9)
    assert sum(result["action_counts"].values()) == result["executed_ticks"]
    assert sum(result["loss_by_action"].values()) == pytest.approx(result["action_opportunity_loss"])
    assert sum(sum(p["loss_by_action"].values()) for p in result["phases"].values()) == pytest.approx(
        result["action_opportunity_loss"])
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("initial,budget,actions,speeds", [
    (10, 3, ["ACCELERATE"] * 3, [10.1, 10.2, 10.3]),
    (10, 3, ["NOTHING"] * 3, [10, 10, 10]),
    (0.05, 4, ["DECELERATE"] * 3 + ["ACCELERATE"], [0, 0, 0, 0.1]),
    (0.05, 7, ["DECELERATE", "ACCELERATE"], [0, 0.1]),
    (10, 9, ["STEER_LEFT", "STEER_RIGHT", "NOTHING"], [10, 10, 10]),
    (10, 5, [], []),
    (10, 0, [], []),
])
def test_distance_identity_including_clipping_and_termination(initial, budget, actions, speeds):
    result = account(initial, budget, actions, speeds)
    assert_identity(result)
    assert result["ceiling"] == pytest.approx(initial * budget + 0.05 * budget * (budget + 1))
    remaining = budget - len(speeds)
    final_speed = speeds[-1] if speeds else initial
    assert result["termination_loss"] == pytest.approx(sum(final_speed + 0.1 * j
                                                         for j in range(1, remaining + 1)))


def test_clipping_uses_actual_delta_not_nominal_braking():
    result = account(0.05, 3, ["DECELERATE"] * 3, [0, 0, 0])
    assert result["loss_by_action"]["DECELERATE"] == pytest.approx(0.15 * 3 + 0.1 * 2 + 0.1)
    assert result["termination_loss"] == 0


def test_phase_and_action_groups_use_full_budget_not_executed_length():
    actions = ["NOTHING", "DECELERATE", "STEER_LEFT", "STEER_RIGHT", "ACCELERATE", "NOTHING"]
    result = account(10, 9, actions, [10, 9.9, 9.9, 9.9, 10, 10])
    assert_identity(result)
    early, middle, late = (result["phases"][name] for name in ("early", "middle", "late"))
    assert early["action_counts"] == {"NOTHING": 1, "DECELERATE": 1, "STEER_LEFT": 1}
    assert early["loss_by_action"] == pytest.approx({"NOTHING": 0.9, "DECELERATE": 1.6, "STEER_LEFT": 0.7})
    assert middle["action_counts"] == {"STEER_RIGHT": 1, "ACCELERATE": 1, "NOTHING": 1}
    assert middle["loss_by_action"] == pytest.approx({"STEER_RIGHT": 0.6, "ACCELERATE": 0, "NOTHING": 0.4})
    assert late == {"action_counts": {}, "loss_by_action": {}}
    assert result["action_counts"]["NOTHING"] == 2


@pytest.mark.parametrize("budget,expected", [(1, [1, 0, 0]), (2, [1, 1, 0]), (5, [2, 2, 1]), (6, [2, 2, 2])])
def test_third_boundaries_for_small_and_uneven_budgets(budget, expected):
    result = account(10, budget, ["NOTHING"] * budget, [10] * budget)
    assert [sum(phase["action_counts"].values()) for phase in result["phases"].values()] == expected


def test_plan_fields_are_optional_decision_counts_not_inferred_vetoes():
    diagnostics = DistanceDiagnostics(10, 30)
    plans = [
        {"recovery_checks": 4, "recovery_safe": True, "safe_ticks": 8, "horizon": 8, "target": 375},
        {"recovery_checks": 2, "recovery_safe": False, "safe_ticks": 3, "horizon": 8, "target": 599},
        {"safe_ticks": 0, "target": 599},
        {},
        {"target": 375},
    ]
    original = json.dumps(plans)
    for plan in plans:
        diagnostics.record_decision(plan)
    counts = diagnostics.result(0)["decision_counts"]
    assert counts["decisions"] == 5 and counts["decisions_with_plan"] == 4
    assert counts["recovery_checks_sum"] == 6 and counts["recovery_checks_reported_decisions"] == 2
    assert counts["recovery_safe_decisions"] == 1 and counts["recovery_unsafe_decisions"] == 1
    assert counts["safe_ticks_reported_decisions"] == 3
    assert counts["horizon_reported_decisions"] == counts["safe_horizon_comparable_decisions"] == 2
    assert counts["selected_short_horizon_decisions"] == counts["selected_zero_safe_ticks_decisions"] == 1
    assert counts["target_reported_decisions"] == 4
    assert counts["target_comparable_decisions"] == 2 and counts["target_switch_decisions"] == 1
    assert not any("veto" in key or "blocked" in key for key in counts)
    assert json.dumps(plans) == original


def test_additive_future_fields_only_when_reported():
    diagnostics = DistanceDiagnostics(10, 10)
    diagnostics.record_decision({"recovery_vetoes": 3, "observed_blocked": 0, "unseen_blocked": 2})
    diagnostics.record_decision({"recovery_vetoes": 1, "observed_blocked": 4})
    counts = diagnostics.result(0)["decision_counts"]
    for field, total, reported, positive in [("recovery_vetoes", 4, 2, 2),
                                               ("observed_blocked", 4, 2, 1),
                                               ("unseen_blocked", 2, 1, 1)]:
        assert counts[f"{field}_sum"] == total
        assert counts[f"{field}_reported_decisions"] == reported
        assert counts[f"{field}_positive_decisions"] == positive


@pytest.mark.parametrize("crash_at", [None, 3])
def test_episode_counts_executed_actions_and_decisions_with_clipping(game, monkeypatch, crash_at):
    initialize, original_step = core.initialize_game_state, benchmark.step

    def low_speed(*args, **kwargs):
        initialize(*args, **kwargs)
        core.STATE.ego.velocity.x = 0.05

    class ScriptedDriver:
        def __init__(self, config):
            self.last_plan = {}

        def actions(self, obs):
            assert set(obs) == {"did_crash", "elapsed_ticks", "distance", "velocity", "sensors"}
            self.last_plan = {"recovery_checks": 2, "recovery_safe": False, "safe_ticks": 0, "horizon": 8}
            return ["DECELERATE"] * 8

    def step(action):
        state = original_step(action)
        if state.ticks == crash_at:
            state.crashed = True
        return state

    monkeypatch.setattr(core, "initialize_game_state", low_speed)
    monkeypatch.setattr(benchmark, "ExpertController", ScriptedDriver)
    monkeypatch.setattr(benchmark, "step", step)
    times = iter([0, 0.001, 1, 1.009])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    result = benchmark.run_episode("distance-counts", max_ticks=10)
    diagnostics = result["diagnostics"]
    assert_identity(diagnostics)
    assert diagnostics["action_counts"] == {"DECELERATE": crash_at or 10}
    assert diagnostics["decision_counts"]["decisions"] == result["requests"] == (1 if crash_at else 2)
    assert diagnostics["decision_counts"]["recovery_checks_sum"] == 2 * result["requests"]
    assert diagnostics["remaining_ticks"] == (7 if crash_at else 0)
    assert result["crashed"] == bool(crash_at)
    expected = np.percentile([1] if crash_at else [1, 9], [95, 99])
    assert result["p95_decision_ms"] == pytest.approx(expected[0])
    assert result["p99_decision_ms"] == pytest.approx(expected[1])
    assert "trace" not in result


def test_empty_episode_latency_is_safe(game):
    result = benchmark.run_episode("distance-empty", max_ticks=0)
    for key in ("requests", "mean_decision_ms", "max_decision_ms", "p95_decision_ms", "p99_decision_ms"):
        assert result[key] == 0
    assert result["diagnostics"]["decision_counts"]["decisions"] == 0
    assert_identity(result["diagnostics"])


@pytest.mark.parametrize("policy,planner", [("accelerate", "legacy"), ("coast", "legacy"),
                                           ("expert", "legacy"), ("expert", "maneuver")])
def test_diagnostics_preserve_policy_observations_physics_and_rng(game, monkeypatch, policy, planner):
    config = Config(planner=planner)
    seed, budget = "distance-invariance", 24
    core.initialize_game_state("", seed)
    driver, queue = ExpertController(config), deque()
    expected_observations, expected_actions = [], []
    while core.STATE.ticks < budget and not core.STATE.crashed:
        if not queue:
            obs = benchmark.observation(core.STATE)
            expected_observations.append(obs)
            queue.extend(driver.actions(obs) if policy == "expert" else
                         ["ACCELERATE" if policy == "accelerate" else "NOTHING"] * 8)
        action = queue.popleft()
        expected_actions.append(action)
        benchmark.step(action)

    def snapshot():
        state = core.STATE
        return (state.ticks, state.distance, state.crashed, state.ego.velocity.x, state.ego.velocity.y,
                [(car.x, car.y, car.velocity.x, car.velocity.y) for car in state.cars],
                randomizer.rng.getstate())

    expected_state = snapshot()
    actual_observations, actual_actions = [], []
    original_observation, original_step = benchmark.observation, benchmark.step

    def observe(state):
        obs = original_observation(state)
        actual_observations.append(obs)
        return obs

    def step(action):
        actual_actions.append(action)
        return original_step(action)

    monkeypatch.setattr(benchmark, "observation", observe)
    monkeypatch.setattr(benchmark, "step", step)
    result = benchmark.run_episode(seed, policy=policy, config=config, max_ticks=budget, trace=True)
    assert actual_observations == expected_observations
    assert actual_actions == expected_actions
    assert snapshot() == expected_state
    assert len(result["trace"]) == result["requests"] == len(expected_observations)
    assert_identity(result["diagnostics"])


def test_summary_500k_crashes_fail_and_existing_keys_remain():
    results = [{"distance": distance, "crashed": crashed, "ticks": 3600, "mean_decision_ms": 2}
               for distance, crashed in [(500000, False), (600000, True), (499999, False), (510000, False)]]
    summary = benchmark.summarize(results)
    assert summary["success_500k_count"] == 2 and summary["success_500k_rate"] == 0.5
    assert summary["p10_distance"] == pytest.approx(499999.3)
    assert summary["games"] == 4 and summary["survived"] == 3
    assert summary["mean_distance"] == 527499.75 and summary["median_distance"] == 505000
    assert summary["min_distance"] == 499999 and summary["max_distance"] == 600000
    assert summary["mean_decision_ms"] == 2
    results[0]["diagnostics"] = {"actual_distance": 499999.9996}
    assert benchmark.summarize(results)["success_500k_count"] == 1


def test_empty_summary_is_safe():
    assert all(value == 0 for value in benchmark.summarize([]).values())