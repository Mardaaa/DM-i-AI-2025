import math

from pydantic import BaseModel, Field, field_validator
from typing import Dict, Optional, List


class RaceCarPredictRequestDto(BaseModel):
    did_crash: bool
    elapsed_ticks: int = Field(ge=0)
    distance: float = Field(ge=0, allow_inf_nan=False)
    velocity: Dict[str, float]  
    sensors: Dict[str, Optional[float]] 

    @field_validator("velocity")
    @classmethod
    def valid_velocity(cls, value):
        if not {"x", "y"} <= value.keys() or not all(math.isfinite(v) for v in value.values()):
            raise ValueError("velocity must contain finite x and y components")
        if value["x"] < 0:
            raise ValueError("forward velocity cannot be negative")
        return value

    @field_validator("sensors")
    @classmethod
    def valid_sensors(cls, value):
        if any(v is not None and (not math.isfinite(v) or not 0 <= v <= 1000.000001)
               for v in value.values()):
            raise ValueError("sensor readings must be null or finite distances from 0 to 1000")
        return value


class RaceCarPredictResponseDto(BaseModel):
    actions: List[str]
    # 'ACCELERATE'
    # 'DECELERATE'
    # 'STEER_LEFT'
    # 'STEER_RIGHT'
    # 'NOTHING''
