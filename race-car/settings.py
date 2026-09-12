"""Deployment preset, separate from the immutable benchmark baseline defaults."""

import os
from pathlib import Path

from expert import Config, load_config


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "expert.json"


def driver_config() -> Config:
    """RACE_CAR_CONFIG overrides the selected preset; invalid files fail fast."""
    return load_config(os.environ.get("RACE_CAR_CONFIG", str(DEFAULT_CONFIG)))