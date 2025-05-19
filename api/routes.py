from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
import datetime
import asyncio
import logging
import numpy as np

from app import (
    SimulationManager, get_manager_dependency, ControlState,
    DEFAULT_START_YEAR, DEFAULT_START_MONTH, DEFAULT_START_DAY, DEFAULT_START_HOUR,
    REAL_TIME_STEP_SEC
)

from .models import (
    BaseResponse, ControlModeRequest, ManualControlsRequest, SimulatorParamsRequest,
    BatchSimulationRequest, SimulationStartRequest, SnapshotResponse, CityChangeResponse,
    SimulationControlResponse, ManualControlsResponse, SimulatorParamsResponse,
    BatchSimulationResponse, WebSocketMessage, WebSocketError
)

# --- API Router Setup ---
router = APIRouter(
    prefix="/api",
    tags=["simulation"],
    responses={404: {"description": "Not found"}},
)

# --- Helper Functions ---
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(i) for i in obj)
    elif isinstance(obj, (np.bool_, np.integer, np.floating)):
        return obj.item()
    return obj

# --- API Endpoints ---
@router.get("/snapshot", response_model=SnapshotResponse)
async def get_snapshot(current_manager: SimulationManager = Depends(get_manager_dependency)):
    """Get the current simulation snapshot."""
    return {
        "status": "success",
        "data": current_manager.get_snapshot_json()
    }

@router.post("/set_city/{new_city}", response_model=CityChangeResponse)
async def set_city(
    new_city: str,
    current_manager: SimulationManager = Depends(get_manager_dependency)
):
    """Change the simulation city."""
    try:
        current_manager.set_city_and_reload(new_city)
        return {
            "status": "success",
            "detail": "City change successful",
            "data": {"city": new_city}
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        current_manager.logger.error(f"API /set_city error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@router.post("/simulation/start", response_model=SimulationControlResponse)
async def start_simulation(
    payload: Optional[SimulationStartRequest] = None,
    current_manager: SimulationManager = Depends(get_manager_dependency)
):
    """Start the simulation with optional start time."""
    current_manager.start_simulation(start_datetime_payload=payload.dict() if payload else None)
    return {
        "status": "success",
        "detail": "Simulation started",
        "data": {
            "start_time": current_manager.simulator.current_datetime.isoformat() if current_manager.simulator else "N/A"
        }
    }

@router.post("/simulation/stop", response_model=SimulationControlResponse)
async def stop_simulation(
    current_manager: SimulationManager = Depends(get_manager_dependency)
):
    """Stop the simulation."""
    current_manager.stop_simulation()
    return {
        "status": "success",
        "detail": "Simulation stopped"
    }

@router.post("/simulation/set_control_mode", response_model=SimulationControlResponse)
async def set_control_mode(
    payload: ControlModeRequest,
    current_manager: SimulationManager = Depends(get_manager_dependency)
):
    """Set the control mode for the simulation."""
    if not current_manager.set_control_mode(payload.control_mode):
        raise HTTPException(status_code=400, detail=f"Invalid control mode: {payload.control_mode}")
    return {
        "status": "success",
        "detail": "Control mode set",
        "data": {"new_mode": payload.control_mode}
    }

@router.post("/actuators/manual_set", response_model=ManualControlsResponse)
async def set_manual_actuators(
    payload: ManualControlsRequest,
    current_manager: SimulationManager = Depends(get_manager_dependency)
):
    """Set manual actuator controls."""
    try:
        manual_cs = ControlState(**payload.dict())
        current_manager.set_manual_overrides(manual_cs)
        return {
            "status": "success",
            "detail": "Manual overrides set",
            "data": manual_cs.to_dict()
        }
    except TypeError as e:
        current_manager.logger.error(f"Invalid payload for manual_set: {payload} - Error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid payload format for manual actuators: {e}")
    except Exception as e:
        current_manager.logger.error(f"Error in /manual_set: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error setting manual overrides: {e}")

@router.post("/reset", response_model=SimulationControlResponse)
async def reset_simulation(
    payload: Optional[SimulationStartRequest] = None,
    current_manager: SimulationManager = Depends(get_manager_dependency)
):
    """Reset the simulation with optional start time."""
    start_dt = None
    if payload and payload.start_datetime:
        start_dt = payload.start_datetime
    else:
        try:
            start_dt = datetime.datetime(
                DEFAULT_START_YEAR,
                DEFAULT_START_MONTH,
                DEFAULT_START_DAY,
                DEFAULT_START_HOUR
            )
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid start_datetime payload for reset: {e}")

    current_manager.reset_simulation(start_datetime=start_dt)
    return {
        "status": "success",
        "detail": "Simulation reset",
        "data": {
            "new_start_time": current_manager.simulator.current_datetime.isoformat() if current_manager.simulator else "N/A"
        }
    }

@router.post("/simulation/set_simulator_params", response_model=SimulatorParamsResponse)
async def set_simulator_params(
    params: SimulatorParamsRequest,
    current_manager: SimulationManager = Depends(get_manager_dependency)
):
    """Set simulator parameters."""
    try:
        current_manager.set_user_simulator_params(params.dict())
        return {
            "status": "success",
            "detail": "User simulator parameters received. Reset simulation for changes to take full effect.",
            "data": current_manager.user_simulator_params
        }
    except Exception as e:
        current_manager.logger.error(f"API Error setting user simulator params: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error setting user simulator parameters: {e}")

@router.post("/simulation/run_batch", response_model=BatchSimulationResponse)
async def run_batch_simulation(
    payload: BatchSimulationRequest,
    current_manager: SimulationManager = Depends(get_manager_dependency)
):
    """Run a batch simulation."""
    try:
        results = current_manager.run_batch_simulation(
            duration_hours=payload.duration_hours,
            target_control_mode=payload.target_control_mode
        )
        if results.get("status") == "error":
            raise HTTPException(status_code=400, detail=results.get("detail", "Batch run error"))

        sanitized_results = convert_numpy_types(results)
        return {
            "status": "success",
            "data": sanitized_results
        }
    except HTTPException:
        raise
    except Exception as e:
        current_manager.logger.error(f"API Error during batch simulation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error running batch simulation: {e}")

# --- WebSocket Endpoint ---
@router.websocket("/ws/snapshot")
async def ws_snapshot(ws: WebSocket):
    """WebSocket endpoint for real-time simulation updates."""
    await ws.accept()
    logger = logging.getLogger("ws_snapshot")
    logger.info("WebSocket client connected.")
    
    try:
        while True:
            if manager and manager.simulator:
                await ws.send_json(manager.get_snapshot_json())
            else:
                error_msg = WebSocketError(
                    error="Sim not ready.",
                    sim_is_running=False,
                    config={"control_mode": "off"}
                )
                await ws.send_json(error_msg.dict())
            await asyncio.sleep(REAL_TIME_STEP_SEC)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        try:
            await ws.close(code=1011)
        except Exception:
            pass 