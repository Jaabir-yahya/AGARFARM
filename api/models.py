from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
import datetime

# --- Base Models ---
class BaseResponse(BaseModel):
    status: str
    detail: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

# --- Request Models ---
class ControlModeRequest(BaseModel):
    control_mode: str

    @validator('control_mode')
    def validate_control_mode(cls, v):
        valid_modes = {"off", "manual", "ml_normal", "ml_eco", "baseline_normal", "baseline_eco"}
        if v not in valid_modes:
            raise ValueError(f"Invalid control mode. Must be one of: {valid_modes}")
        return v

class ManualControlsRequest(BaseModel):
    fan: bool = Field(False, description="Fan control state")
    ac: bool = Field(False, description="Air conditioning control state")
    vent: bool = Field(False, description="Ventilation control state")
    irrigation: bool = Field(False, description="Irrigation control state")

class SimulatorParamsRequest(BaseModel):
    rain_intensity_multiplier: float = Field(1.0, ge=0.0, description="Rain intensity multiplier")
    rain_probability_multiplier: float = Field(1.0, ge=0.0, description="Rain probability multiplier")
    plant_transpiration_multiplier: float = Field(1.0, ge=0.0, description="Plant transpiration multiplier")
    soil_drying_multiplier: float = Field(1.0, ge=0.0, description="Soil drying multiplier")

class BatchSimulationRequest(BaseModel):
    duration_hours: int = Field(..., gt=0, description="Duration of the batch simulation in hours")
    target_control_mode: str = Field(..., description="Control mode to use (e.g., 'ml_normal', 'baseline_normal')")

    @validator('target_control_mode')
    def validate_control_mode(cls, v):
        valid_modes = {"ml_normal", "ml_eco", "baseline_normal", "baseline_eco"}
        if v not in valid_modes:
            raise ValueError(f"Invalid control mode for batch simulation. Must be one of: {valid_modes}")
        return v

class SimulationStartRequest(BaseModel):
    start_datetime: Optional[datetime.datetime] = Field(None, description="Start datetime for the simulation")

# --- Response Models ---
class SnapshotResponse(BaseResponse):
    data: Dict[str, Any] = Field(..., description="Current simulation snapshot data")

class CityChangeResponse(BaseResponse):
    data: Dict[str, str] = Field(..., description="City change information")

class SimulationControlResponse(BaseResponse):
    data: Optional[Dict[str, Any]] = Field(None, description="Simulation control information")

class ManualControlsResponse(BaseResponse):
    data: Dict[str, bool] = Field(..., description="Current manual control states")

class SimulatorParamsResponse(BaseResponse):
    data: Dict[str, float] = Field(..., description="Current simulator parameters")

class BatchSimulationResponse(BaseResponse):
    data: Dict[str, Any] = Field(..., description="Batch simulation results")

# --- WebSocket Models ---
class WebSocketMessage(BaseModel):
    type: str = Field(..., description="Message type")
    data: Optional[Dict[str, Any]] = Field(None, description="Message data")

class WebSocketError(BaseModel):
    error: str = Field(..., description="Error message")
    sim_is_running: bool = Field(False, description="Simulation running state")
    config: Dict[str, str] = Field(..., description="Current configuration") 