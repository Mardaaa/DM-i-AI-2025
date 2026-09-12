import json

import pytest

from expert import Config, load_config
from optuna_search import MAX_DISTANCE, lower_quantile, metrics, selection_key, wilson_interval


def result(distance, crashed=False):
    return {"distance": distance, "crashed": crashed, "ticks": 1000 if crashed else 3600,
            "mean_decision_ms": 1.0}


def test_theoretical_maximum_is_below_one_million():
    assert MAX_DISTANCE == 684180
    assert sum(10 + 0.1 * t for t in range(1, 3601)) == pytest.approx(MAX_DISTANCE)


def test_crashes_penalized_and_all_crashes_zero_utility():
    safe = metrics([result(200000), result(200000)])
    fragile = metrics([result(200000), result(200000, True)])
    assert fragile["robust_utility"] == pytest.approx(safe["robust_utility"] / 4)
    assert metrics([result(200000, True)])["robust_utility"] == 0


def test_tail_quantile_and_wilson_interval():
    assert lower_quantile([100, 200, 300]) == pytest.approx(120)
    assert lower_quantile([100]) == 100
    low, high = wilson_interval(100, 100)
    assert 0.96 < low < 0.97
    assert high == pytest.approx(1)


def test_selection_enforces_survival_floor():
    safe = {"summary": metrics([result(150000)] * 100)}
    risky = {"summary": metrics([result(400000)] * 90 + [result(10000, True)] * 10)}
    assert selection_key(safe, 0.96) > selection_key(risky, 0.96)


@pytest.mark.parametrize("wrapped", [False, True])
def test_config_json_roundtrip(tmp_path, wrapped):
    from dataclasses import asdict
    config = Config(horizon=150, blind_speed_buffer=3.0, lane_change_penalty=0.2)
    data = {"config": asdict(config)} if wrapped else asdict(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    assert load_config(path) == config


@pytest.mark.parametrize("kwargs", [{"blind_speed_buffer": -1}, {"lane_change_penalty": float("inf")},
                                    {"stale_velocity_uncertainty": float("nan")}])
def test_new_config_fields_validated(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_deployed_preset_matches_optuna_selection(monkeypatch):
    from settings import DEFAULT_CONFIG, driver_config
    monkeypatch.delenv("RACE_CAR_CONFIG", raising=False)
    selected = DEFAULT_CONFIG.parents[1] / "results" / "optuna-v1" / "selected-config.json"
    assert driver_config() == load_config(selected)


def test_deployment_config_override(tmp_path, monkeypatch):
    from dataclasses import asdict
    from settings import driver_config
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(asdict(Config())))
    monkeypatch.setenv("RACE_CAR_CONFIG", str(path))
    assert driver_config() == Config()