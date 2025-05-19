# AGARFARM - Development Log: Key Issues & Resolutions

This log summarizes significant technical challenges, errors, and conceptual hurdles encountered and resolved during the AGARFARM project development.

---

## I. Initial Simulator & Reinforcement Learning Environment Setup

### Simulator Initialization Discrepancies
**Issue**: Inconsistent parameter names for simulation start time (e.g., `sim_month` vs `start_month`, etc.)
**Resolution**: Standardized `GreenhouseSimulator.__init__` to accept `start_year`, `start_month`, `start_day_of_month`, and `start_hour`. Updated all scripts (`train_ppo.py`, `run_evaluation_engine.py`, `app.py`) accordingly.

---

### NameError in SubprocVecEnv for Global Constants
**Issue**: Subprocesses used during parallel training couldn’t access global constants (`ACTUATORS`, `OBS_DIM`, etc.) in `custom_env_wrapper.py`.
**Resolution**: Ensured constants were defined at the module level in `custom_env_wrapper.py`, imported from canonical sources like `evaluation/scenarios.py`.

---

### Observation Space Dimension Mismatches
**Issue**: Observation dimension definitions (`OBS_DIM`, `BASE_OBS_DIM`) were inconsistent across modules, leading to shape errors during model loading or prediction.
**Resolution**: Centralized definitions in `evaluation/scenarios.py` and ensured consistent imports across training, evaluation, and inference code. Added sanity checks for mismatches.

---

### Actuator History Implementation Bugs
**Issue**: Incorrect logic in generating actuator history features caused mismatched input shapes and out-of-order feature vectors.
**Resolution**: Corrected the history loop in `GreenhouseEnv._get_observation()` and ensured `actuator_state_history_buffer` was updated at the correct simulator stage.

---

### TypeError in Reward Calculation
**Issue**: Reward functions sometimes used `ControlState` objects directly, or encountered `None` values.
**Resolution**: Ensured only primitive types (floats, bools) were used in reward functions. Accessed dataclass attributes properly (e.g., `action.fan`, not `action['fan']`).

---

## II. ML Model Training & Evaluation Pipeline

### NumPy Types Not JSON Serializable
**Issue**: `numpy.bool_` and `numpy.float64` caused errors during JSON serialization in API responses and when saving diagnostics.
**Resolution**: Added `convert_to_python_types()` helper to recursively cast NumPy types to native Python equivalents before JSON export.

---

### NameError: 'OBS_DIM' Not Defined in Evaluation
**Issue**: `OBS_DIM` used in `run_evaluation_engine.py` without being defined locally.
**Resolution**: Imported `OBSERVATION_FEATURE_ORDER` and derived `OBS_DIM` from its length within the module.

---

### ML Agent Behavior Issues

#### Issue 1: Actuator Chattering
**Problem**: ML agents toggled actuators too frequently.
**Fix**: Introduced `W_ACTION_CHANGE` penalty in reward function. Tuned reward weights.

#### Issue 2: Lack of In-Band Stability
**Problem**: Agents kept making micro-adjustments while already within target bands.
**Fix**: Added "stability bonus" and widened buffer zones to create a deadband.

#### Issue 3: Suboptimal Energy Use
**Problem**: Agents sometimes acted wastefully or illogically.
**Fix**: Penalized sustained out-of-band behavior and introduced "sensible action" guidance in rewards.

---

## III. FastAPI Backend & Frontend (app.py, index.html)

### NameError: 'simulation_control_event' Undefined
**Issue**: `simulation_control_event` used before being defined during global initialization.
**Resolution**: Moved its definition to appear before the `SimulationManager` instance in `app.py`.

---

### AttributeError: Missing SimulationManager Attributes
**Issue**: `self.simulator` or `self.is_running` were accessed even when `SimulationManager.__init__()` failed partway.
**Resolution**: Fixed root cause (e.g., failed snapshot initialization), added guards to ensure attributes were present.

---

### CORS Errors

#### Issue 1: Blocked by Browser
**Problem**: The frontend could not reach the backend due to CORS restrictions.
**Fix**: Configured `CORSMiddleware` in `app.py` to allow specific origins or `*` for debugging.

#### Issue 2: Preflight Failures
**Problem**: Failing OPTIONS requests due to startup errors preventing CORS headers from being added.
**Fix**: Fixed initialization-time backend exceptions (e.g., `SensorSnapshot` TypeErrors) to allow proper CORS behavior.

---

### WebSocket Connection Failures

#### Issue 1: Incorrect Path
**Problem**: Frontend connected to `/ws` instead of `/ws/snapshot`.
**Fix**: Corrected `WS_BASE_URL` in `index.html`.

#### Issue 2: Backend Crash Mid-WebSocket
**Problem**: `SensorSnapshot.__init__()` threw TypeError due to missing or unexpected args, causing the WebSocket to close.
**Fix**: Updated `SensorSnapshot` dataclass to accept optional `outside_temp`, `outside_rh`. Fixed field names and defaults.

---

### UI State Overwritten by Incoming WebSocket Data
**Issue**: User inputs (sliders, dropdowns, checkboxes) were reset immediately by incoming state updates.
**Resolution**: Updated `socket.onmessage` to only update UI fields if not currently focused and values differ. Manual actuators now fully user-driven.

---

### TypeError: SensorSnapshot Init
**Issue**: `SensorSnapshot` was initialized with fields (`outside_temp`, etc.) that didn’t exist.
**Resolution**: Updated `simulator/core.py` to match latest version, ensured all field names and defaults matched app expectations.

---

## IV. Deployment

### Auto-Suspension and WebSocket Drops
**Issue**: The app randomly stopped accepting WebSocket connections after periods of inactivity.
**Resolution**: Added logging and monitoring. Ensured app restarts cleanly and WebSocket reconnect logic is in place.

---

### Missing or Incorrect `requirements.txt` or Startup Command
**Issue**: Server failed to start due to bad dependencies or wrong Uvicorn command.
**Resolution**: Regenerated `requirements.txt` using `pip freeze`. Ensured startup command was:
```bash
uvicorn app:app --host 0.0.0.0 --port $PORT