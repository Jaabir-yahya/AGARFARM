import os
import threading
import time
import math
import asyncio
import yaml
import numpy as np
import collections
import datetime
import logging
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from typing import Dict, Any, Optional, List

# --- Basic Logging Config (place early) ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s')
logger = logging.getLogger("AGARFARM_APP")

# --- Project Path Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Attempt to Import Project Modules ---
try:
    from simulator.core import (
        GreenhouseSimulator, SensorSnapshot, ControlState, vapor_pressure_deficit,
        TEMP_FLOOR, TEMP_CAP, RH_TERMINATE_MIN, RH_TERMINATE_MAX, SM_TERMINATE_MIN
    )
    from controllers.smart_ml_agent import SmartMLAgent
    from controllers.baseline import BaselineController
    from evaluation.scenarios import (
        OBSERVATION_FEATURE_ORDER as SCENARIOS_OBS_ORDER,
        N_ACTUATOR_HISTORY_STEPS as SCENARIOS_N_HIST_STEPS,
        ACTUATORS as SCENARIOS_ACTUATORS,
        BASE_OBSERVATION_FEATURE_ORDER as SCENARIOS_BASE_OBS_ORDER
    )
    # This import is for the reward calculation logic 
    from ml_training.reward_functions import calculate_reward

    logger.info("Successfully imported project modules in app.py")
    OBSERVATION_FEATURE_ORDER = SCENARIOS_OBS_ORDER
    BASE_OBSERVATION_FEATURE_ORDER = SCENARIOS_BASE_OBS_ORDER
    N_ACTUATOR_HISTORY_STEPS = SCENARIOS_N_HIST_STEPS
    ACTUATORS = SCENARIOS_ACTUATORS
    N_ACTUATORS = len(ACTUATORS)
    BASE_OBS_DIM = len(BASE_OBSERVATION_FEATURE_ORDER)
    if len(OBSERVATION_FEATURE_ORDER) != 28:
        logger.warning(f"Imported OBSERVATION_FEATURE_ORDER length is {len(OBSERVATION_FEATURE_ORDER)}, expected 28!")
except ImportError as e:
    logger.error(f"CRITICAL ERROR: Failed to import project modules in app.py: {e}", exc_info=True)
    logger.warning("Using placeholder classes/constants. Functionality WILL BE AFFECTED.")

    # Define placeholder classes and constants
    class GreenhouseSimulator:
        def __init__(self, *args, **kwargs): 
            self.logger = logging.getLogger("DummySim")
            self.logger.error("DummySim Used!")
            raise NotImplementedError("Sim not loaded")

        def get_actual_actuator_states(self): 
            return ControlState()

        def _is_currently_raining(self): 
            return False

        def get_outside_temp(self): 
            return 20.0
        get_outside_rh = lambda s: 50.0

        def resource_totals(self): 
            return {"kwh": 0.0, "water_l": 0.0}

        current_datetime = datetime.datetime.now()
        temp_c, rh, sm, rain_tank = 20, 50, 50, 100

        def step(self, *args, **kwargs): 
            return SensorSnapshot(
                self.temp_c, self.rh, self.sm, 0.0, self.rain_tank,
                self.current_datetime.isoformat(), 20.0, 50.0
            )

    class SensorSnapshot:
        def __init__(self, tc, rh, sm, vpd, rt, dt_iso, outside_temp=None, outside_rh=None):
            self.t_c = tc
            self.rh = rh
            self.sm = sm
            self.vpd = vpd
            self.rain_tank_l = rt
            self.current_datetime_iso = dt_iso
            self.outside_temp = outside_temp
            self.outside_rh = outside_rh

        def to_dict(self): 
            return {k: v for k, v in self.__dict__.items()}

    class ControlState:
        def __init__(self, fan=False, ac=False, vent=False, irrigation=False): 
            self.fan = fan
            self.ac = ac
            self.vent = vent
            self.irrigation = irrigation

        def to_dict(self): 
            return {
                "fan": self.fan,
                "ac": self.ac,
                "vent": self.vent,
                "irrigation": self.irrigation
            }
        copy = lambda s: ControlState(**s.__dict__)

    class SmartMLAgent:
        def __init__(self, *args, **kwargs): 
            self.model = None
            self.model_name = "NoModelLoaded"
            self.model_metadata = {}
            self.logger = logging.getLogger("DummyAgent")
            self.logger.error("DummyAgent Used!")

        def get_action(self, obs): 
            return ControlState()

    class BaselineController:
        def __init__(self, *args, **kwargs): 
            self.logger = logging.getLogger("DummyBaseline")
            self.logger.error("DummyBaseline Used!")

        def get_controls(self, *args, **kwargs): 
            return ControlState()

    def vapor_pressure_deficit(t, rh):
        return 0.0

    def calculate_reward(*args, **kwargs):
        return 0.0  # Placeholder

    OBSERVATION_FEATURE_ORDER = []
    BASE_OBSERVATION_FEATURE_ORDER = []
    N_ACTUATOR_HISTORY_STEPS = 3
    ACTUATORS = ['fan', 'ac', 'vent', 'irrigation']
    N_ACTUATORS = len(ACTUATORS)
    BASE_OBS_DIM = 16
    TEMP_FLOOR, TEMP_CAP = 0.0, 50.0
    RH_TERMINATE_MIN, RH_TERMINATE_MAX, SM_TERMINATE_MIN = 5.0, 100.0, 0.0

# --- Global Constants ---
CITY_CONFIG_DIR = os.path.join(PROJECT_ROOT, "simulator", "city_configs")
MODEL_DIR = os.path.join(PROJECT_ROOT, "evaluation", "trained_models")
HPARAMS_DIR = os.path.join(PROJECT_ROOT, "ml_training", "config")  # Path to hparams YAMLs
SIM_DT_MIN = 5
REAL_TIME_STEP_SEC = 1
MAX_HISTORY_LENGTH = 24
DEFAULT_START_YEAR = 2025
DEFAULT_START_MONTH = 7
DEFAULT_START_DAY = 1
DEFAULT_START_HOUR = 6
OVERLAY = {
    ("oslo", "normal"): "o_normal",
    ("oslo", "eco"): "o_eco",
    ("riyadh", "normal"): "r_normal",
    ("riyadh", "eco"): "r_eco"
}
VALID_CITIES = {"oslo", "riyadh"}
VALID_CONTROL_MODES = {"off", "manual", "ml_normal", "ml_eco", "baseline_normal", "baseline_eco"}
CRITICAL_TEMP_MIN = -5.0
CRITICAL_TEMP_MAX = 55.0
CRITICAL_RH_MIN = 5.0
CRITICAL_RH_MAX = 105.0

# --- Import API Modules ---
from api.routes import router as api_router
from api.errors import (
    SimulationError, ConfigurationError, ControlError, ResourceError,
    simulation_error_handler, http_exception_handler, general_exception_handler
)
from api.docs import custom_openapi, API_TITLE, API_VERSION, API_DESCRIPTION, API_TAGS

# --- FastAPI App Setup ---
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    openapi_tags=API_TAGS
)

# Set custom OpenAPI schema
app.openapi = lambda: custom_openapi(app)

ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://kzmkgaq2z64wzepycz0n.lite.vusercontent.net",
    "https://agarfarm-api.onrender.com",
    "https://agarfarm.onrender.com"  # Add your frontend Render URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Use the list instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register error handlers
app.add_exception_handler(SimulationError, simulation_error_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Include API router
app.include_router(api_router)

# --- SimulationManager Class Definition ---
class SimulationManager:
    def __init__(self, initial_city: str = "oslo"):
        self.logger = logging.getLogger(f"SimManager.{initial_city}")
        self.lock = threading.Lock()
        self.city: str = initial_city.lower()
        if self.city not in VALID_CITIES:
            self.logger.error(f"Invalid city: {self.city}. Defaulting.")
            self.city = "oslo"
        self.city_config_path: str = os.path.join(CITY_CONFIG_DIR, f"{self.city}.yaml")
        self.city_config_data: Optional[Dict[str, Any]] = None
        self.ml_training_target_ranges: Dict[str, float] = {}
        self.current_reward_params: Dict[str, float] = {}  # Store reward params for current ML agent

        self.is_running: bool = False
        self.current_control_mode: str = "off"
        self.manual_actuator_overrides: ControlState = ControlState()
        self.simulation_error_message: Optional[str] = None
        self.user_display_target_ranges: Optional[Dict[str, float]] = None
        self.user_simulator_params: Dict[str, float] = {
            "rain_intensity_multiplier": 1.0,
            "rain_probability_multiplier": 1.0,
            "plant_transpiration_multiplier": 1.0,
            "soil_drying_multiplier": 1.0
        }

        # Initialize simulator and controllers
        self.simulator: Optional[GreenhouseSimulator] = None
        self.active_controller: Optional[Union[SmartMLAgent, BaselineController]] = None
        self.active_controller_name: str = "none"
        self.active_controller_metadata: Dict[str, Any] = {}

        # Load initial configuration
        self._load_city_config()
        self._load_all_controllers_for_city()
        self._init_simulator()

    def _load_city_config(self):
        self.city_config_path = os.path.join(CITY_CONFIG_DIR, f"{self.city}.yaml")
        if not os.path.exists(self.city_config_path): raise ValueError(
            f"City config not found: {self.city_config_path}")
        with open(self.city_config_path) as f: self.city_config_data = yaml.safe_load(f)
        self.ml_training_target_ranges = {
            "t_c_min": self.city_config_data["default_target_t_c_min"],
            "t_c_max": self.city_config_data["default_target_t_c_max"],
            "rh_min": self.city_config_data["default_target_rh_min"],
            "rh_max": self.city_config_data["default_target_rh_max"],
            "sm_min": self.city_config_data["default_target_sm_min"],
            "sm_max": self.city_config_data["default_target_sm_max"],
        }
        self.logger.info(f"Loaded city config for {self.city}. ML Targets: {self.ml_training_target_ranges}")
        # Also try to load default reward params for the city if ML mode is selected later

    def _load_reward_params_for_agent(self, mode_suffix: str) -> Dict[str, float]:
        """Loads reward parameters from the hparams file for a given city and mode."""
        hparams_path = os.path.join(HPARAMS_DIR, f"{self.city}_{mode_suffix}_hparams.yaml")
        if os.path.exists(hparams_path):
            with open(hparams_path, 'r') as f: hparams_data = yaml.safe_load(f)
            return hparams_data.get('reward_params', {})
        self.logger.warning(
            f"HParams file not found for {self.city}_{mode_suffix} at {hparams_path}. Using empty reward_params.")
        return {}

    def _load_agent_instance(self, mode_suffix: str) -> Optional[SmartMLAgent]:
        agent_key = f"{self.city}_{mode_suffix}"
        model_file_suffix = OVERLAY.get((self.city, mode_suffix))
        if not model_file_suffix: self.logger.warning(
            f"No model in OVERLAY for ({self.city}, {mode_suffix})"); return None
        model_path = os.path.join(MODEL_DIR, f"ppo_agent_{model_file_suffix}.zip")
        vec_path = os.path.join(MODEL_DIR, f"vecnormalize_{model_file_suffix}.pkl")
        if not os.path.exists(model_path): self.logger.warning(
            f"Model file not found for {agent_key} at {model_path}"); return None
        self.logger.info(f"Loading agent {agent_key} from {model_path}...")
        vec_arg = vec_path if os.path.exists(vec_path) else None
        try:
            agent = SmartMLAgent(model_path=model_path, vec_normalize_path=vec_arg)
            if agent.model is None: self.logger.error(f"Agent model is None for {agent_key}"); return None
            self.logger.info(f"Loaded agent {agent_key} ({getattr(agent, 'model_name', 'N/A')})")
            return agent
        except Exception as e:
            self.logger.error(f"Exception loading agent {agent_key}: {e}", exc_info=True); return None

    def _load_all_controllers_for_city(self):
        self.logger.info(f"Loading all controllers for city: {self.city}")
        self.agents = {}
        self.baseline_controllers = {}
        self.agents[f"{self.city}_normal"] = self._load_agent_instance("normal")
        self.agents[f"{self.city}_eco"] = self._load_agent_instance("eco")
        if self.city_config_data:
            self.baseline_controllers[f"{self.city}_baseline_normal"] = BaselineController(
                city_config=self.city_config_data, target_ranges=self.ml_training_target_ranges, eco_mode=False)
            self.baseline_controllers[f"{self.city}_baseline_eco"] = BaselineController(
                city_config=self.city_config_data, target_ranges=self.ml_training_target_ranges, eco_mode=True)
            self.logger.info(f"Loaded Baseline controllers for {self.city}")
        else:
            self.logger.error("City config data not loaded, cannot initialize Baseline controllers.")
        self.set_active_controller()  # Renamed

    def set_active_controller(self):
        """Set the active controller based on current control mode."""
        target_controller_key = None
        self.active_controller = None
        self.current_reward_params = {}  # Reset reward params

        try:
            # Determine target controller based on mode
            if self.current_control_mode == "ml_normal":
                target_controller_key = f"{self.city}_normal"
            elif self.current_control_mode == "ml_eco":
                target_controller_key = f"{self.city}_eco"
            elif self.current_control_mode == "baseline_normal":
                target_controller_key = f"{self.city}_baseline_normal"
            elif self.current_control_mode == "baseline_eco":
                target_controller_key = f"{self.city}_baseline_eco"

            if target_controller_key:
                if self.current_control_mode.startswith("ml_"):
                    self.active_controller = self.agents.get(target_controller_key)
                    if self.active_controller:
                        # Validate ML controller
                        if not hasattr(self.active_controller, 'model') or self.active_controller.model is None:
                            raise ConfigurationError(f"ML controller {target_controller_key} has no valid model")
                        # Load reward parameters
                        self.current_reward_params = self._load_reward_params_for_agent(
                            self.current_control_mode.replace("ml_", ""))
                elif self.current_control_mode.startswith("baseline_"):
                    if not self.city_config_data:
                        raise ConfigurationError("City config data not loaded for baseline controller")
                    self.active_controller = self.baseline_controllers.get(target_controller_key)
                    if not self.active_controller:
                        raise ConfigurationError(f"Baseline controller {target_controller_key} not initialized")

                if self.active_controller is None:
                    self.logger.warning(
                        f"Controller for mode {self.current_control_mode} (key: {target_controller_key}) not available/loaded.")
                    raise ConfigurationError(f"Controller not available for mode {self.current_control_mode}")

            # Set controller metadata
            controller_name_display = "None"
            if isinstance(self.active_controller, SmartMLAgent):
                controller_name_display = getattr(self.active_controller, "model_name", "ML Agent")
                self.active_controller_metadata = getattr(self.active_controller, "model_metadata", {})
            elif isinstance(self.active_controller, BaselineController):
                controller_name_display = f"Baseline ({'Eco' if self.active_controller.eco_mode else 'Normal'})"
                self.active_controller_metadata = {"type": "Baseline"}

            self.logger.info(f"Active controller set to: {controller_name_display} (Mode: {self.current_control_mode})")
            if self.current_control_mode.startswith("ml_"):
                self.logger.info(f"  Using reward_params: {self.current_reward_params}")

        except Exception as e:
            self.logger.error(f"Error setting active controller: {e}", exc_info=True)
            raise ConfigurationError(f"Failed to set active controller: {str(e)}")

    def _init_simulator(self, start_datetime: Optional[datetime.datetime] = None):
        self.logger.info(
            f"Initializing simulator. Requested start: {start_datetime}. User params: {self.user_simulator_params}")
        s_year = start_datetime.year if start_datetime else DEFAULT_START_YEAR
        s_month = start_datetime.month if start_datetime else DEFAULT_START_MONTH
        s_day = start_datetime.day if start_datetime else DEFAULT_START_DAY
        s_hour = start_datetime.hour if start_datetime else DEFAULT_START_HOUR
        try:
            self.simulator = GreenhouseSimulator(
                city_config_path=self.city_config_path, init_state=None,
                start_year=s_year, start_month=s_month, start_day_of_month=s_day, start_hour=s_hour,
                rain_intensity_multiplier=self.user_simulator_params.get("rain_intensity_multiplier", 1.0),
                rain_probability_multiplier=self.user_simulator_params.get("rain_probability_multiplier", 1.0),
                plant_transpiration_multiplier=self.user_simulator_params.get("plant_transpiration_multiplier", 1.0),
                soil_drying_multiplier=self.user_simulator_params.get("soil_drying_multiplier", 1.0)
            )
            self.simulator.cumulative_kwh = 0.0;
            self.simulator.cumulative_water_l = 0.0
            initial_outside_temp = self.simulator.get_outside_temp()
            initial_outside_rh = self.simulator.get_outside_rh()
            self.snapshot = SensorSnapshot(
                self.simulator.temp_c, self.simulator.rh, self.simulator.sm,
                vapor_pressure_deficit(self.simulator.temp_c, self.simulator.rh),
                self.simulator.rain_tank, self.simulator.current_datetime.isoformat(),
                outside_temp=initial_outside_temp, outside_rh=initial_outside_rh)
            self.last_requested_controls = self.simulator.get_actual_actuator_states()
            self.sensor_history.clear();
            self.actuator_history.clear()
            initial_ss_dict = self.snapshot.to_dict();
            initial_cs_dict = self.last_requested_controls.to_dict()
            for _ in range(MAX_HISTORY_LENGTH):
                self.sensor_history.append(initial_ss_dict);
                self.actuator_history.append(initial_cs_dict)
            self.simulation_error_message = None
            self.logger.info(f"Simulator initialized. Time: {self.simulator.current_datetime.isoformat()}")
        except Exception as e:
            self.logger.error(f"ERROR during _init_simulator: {e}", exc_info=True)
            self.simulator = None;
            self.snapshot = None

    def step(self):
        """Execute a single simulation step with proper state management."""
        with self.lock:
            if not self.is_running or not self.simulator or self.simulation_error_message:
                return

            try:
                sim = self.simulator
                current_sim_time = sim.current_datetime
                requested_controls_for_step: ControlState

                # Get controls based on current mode
                if self.current_control_mode == "manual":
                    requested_controls_for_step = self.manual_actuator_overrides
                elif self.active_controller:
                    try:
                        # Create a snapshot for the controller
                        current_outside_temp = sim.get_outside_temp()
                        current_outside_rh = sim.get_outside_rh()
                        current_sim_snapshot = SensorSnapshot(
                            sim.temp_c, sim.rh, sim.sm,
                            vapor_pressure_deficit(sim.temp_c, sim.rh),
                            sim.rain_tank, current_sim_time.isoformat(),
                            outside_temp=current_outside_temp, outside_rh=current_outside_rh
                        )

                        if isinstance(self.active_controller, SmartMLAgent):
                            requested_controls_for_step = self._get_ml_controls(current_sim_snapshot, sim)
                        elif isinstance(self.active_controller, BaselineController):
                            requested_controls_for_step = self.active_controller.get_controls(
                                current_sim_snapshot, current_outside_temp, current_outside_rh, 
                                sim._is_currently_raining()
                            )
                        else:
                            self.logger.warning("Unknown controller type, using default controls")
                            requested_controls_for_step = ControlState()
                    except Exception as e:
                        self.logger.error(f"Error getting controller action: {e}", exc_info=True)
                        requested_controls_for_step = ControlState()
                else:
                    requested_controls_for_step = ControlState()

                # Store requested controls
                self.last_requested_controls = requested_controls_for_step

                # Execute simulation step
                try:
                    self.snapshot = sim.step(
                        SIM_DT_MIN, 
                        requested_controls_for_step,
                        (self.ml_training_target_ranges["t_c_min"] + self.ml_training_target_ranges["t_c_max"]) / 2.0
                    )

                    # Check for critical conditions
                    if not (CRITICAL_TEMP_MIN <= self.snapshot.t_c <= CRITICAL_TEMP_MAX):
                        raise ControlError(f"Critical temperature limit exceeded: {self.snapshot.t_c:.1f}°C")
                    if not (CRITICAL_RH_MIN <= self.snapshot.rh <= CRITICAL_RH_MAX):
                        raise ControlError(f"Critical humidity limit exceeded: {self.snapshot.rh:.1f}%")

                except Exception as e:
                    self.simulation_error_message = f"Simulation step error: {str(e)}"
                    self.is_running = False
                    self.logger.error(self.simulation_error_message, exc_info=True)
                    return

                # Update history
                self.actuator_history.append(self.last_requested_controls.to_dict())
                if self.snapshot:
                    self.sensor_history.append(self.snapshot.to_dict())

            except Exception as e:
                self.logger.error(f"Unexpected error in simulation step: {e}", exc_info=True)
                self.simulation_error_message = f"Unexpected error: {str(e)}"
                self.is_running = False

    def _get_ml_controls(self, current_snapshot: SensorSnapshot, sim: GreenhouseSimulator) -> ControlState:
        """Get control actions from ML controller."""
        try:
            hour = sim.current_datetime.hour
            base_obs_dict = {
                't_c': current_snapshot.t_c,
                'rh': current_snapshot.rh,
                'sm': current_snapshot.sm,
                'vpd': current_snapshot.vpd,
                'rain_tank_l': current_snapshot.rain_tank_l,
                'outside_temp': current_snapshot.outside_temp,
                'outside_rh': current_snapshot.outside_rh,
                'sin_hour': math.sin(2 * math.pi * hour / 24.0),
                'cos_hour': math.cos(2 * math.pi * hour / 24.0),
                'is_raining': float(sim._is_currently_raining()),
                **{f"target_{k}": v for k, v in self.ml_training_target_ranges.items()}
            }

            if not OBSERVATION_FEATURE_ORDER or len(OBSERVATION_FEATURE_ORDER) != 28:
                raise ConfigurationError(f"Invalid observation feature order length: {len(OBSERVATION_FEATURE_ORDER)}")

            # Prepare observation vector
            base_values = [base_obs_dict[f] for f in OBSERVATION_FEATURE_ORDER[:BASE_OBS_DIM]]
            history_values = []
            temp_history_list = list(self.actuator_history)
            
            for i in range(1, N_ACTUATOR_HISTORY_STEPS + 1):
                past_control_dict = temp_history_list[-i]
                for act_name in ACTUATORS:
                    history_values.append(float(past_control_dict.get(act_name, False)))

            if len(history_values) != (N_ACTUATORS * N_ACTUATOR_HISTORY_STEPS):
                raise ConfigurationError(f"Invalid history length: {len(history_values)}")

            combined_values = base_values + history_values
            obs_np = np.array(combined_values, dtype=np.float32)
            
            if obs_np.shape != (28,):
                raise ConfigurationError(f"Invalid observation shape: {obs_np.shape}")

            return self.active_controller.get_action(obs_np)

        except Exception as e:
            self.logger.error(f"Error in ML control generation: {e}", exc_info=True)
            return ControlState()

    def get_snapshot_json(self) -> Dict[str, Any]:
        # And ensuring active_controller_name and metadata are correctly populated
        with self.lock:
            sim_time_iso = (self.snapshot.current_datetime_iso if self.snapshot
                            else (
                self.simulator.current_datetime.isoformat() if self.simulator else datetime.datetime.now(
                    datetime.timezone.utc).isoformat() + " (Sim Not Ready)"))
            environment_data = self.snapshot.to_dict() if self.snapshot else {}
            if self.simulator:
                environment_data.setdefault('outside_temp', self.simulator.get_outside_temp())
                environment_data.setdefault('outside_rh', self.simulator.get_outside_rh())

            actual_applied_controls_raw = self.simulator.get_actual_actuator_states().to_dict() if self.simulator else ControlState().to_dict()
            actual_applied_controls = {k: bool(v) for k, v in actual_applied_controls_raw.items()}
            requested_cs_raw = self.last_requested_controls.to_dict() if self.last_requested_controls else ControlState().to_dict()
            requested_cs_dict = {k: bool(v) for k, v in requested_cs_raw.items()}
            resources = self.simulator.resource_totals() if self.simulator else {"kwh": 0.0, "water_l": 0.0}
            is_raining_flag = self.simulator._is_currently_raining() if self.simulator else False

            active_controller_name = "N/A"
            active_controller_metadata = {}
            if isinstance(self.active_controller, SmartMLAgent):
                active_controller_name = getattr(self.active_controller, "model_name", "ML Agent")
                active_controller_metadata = getattr(self.active_controller, "model_metadata", {})
            elif isinstance(self.active_controller, BaselineController):
                active_controller_name = f"Baseline ({'Eco' if self.active_controller.eco_mode else 'Normal'})"
                active_controller_metadata = {"type": "Baseline"}

            current_targets = self.ml_training_target_ranges if self.ml_training_target_ranges else {}
            act_hist_list = [{k: bool(v) for k, v in h_item.items()} for h_item in list(self.actuator_history)]

            config_data = {
                "city": str(self.city), "control_mode": str(self.current_control_mode),
                "sim_is_running": bool(self.is_running), "model_name": active_controller_name,
                "model_metadata": active_controller_metadata, "current_sim_datetime_iso_config": sim_time_iso,
                "user_simulator_params": self.user_simulator_params
            }
            return {
                "environment": environment_data, "controls_applied": actual_applied_controls,
                "controls_requested": requested_cs_dict,
                "resources": {"kwh": float(resources["kwh"]), "water_l": float(resources["water_l"])},
                "config": config_data, "targets": {k: float(v) for k, v in current_targets.items()},
                "user_display_targets": self.user_display_target_ranges,
                "events": {"is_raining": bool(is_raining_flag)},
                "history": {"actuators_requested": act_hist_list, "sensors": list(self.sensor_history)},
                "simulation_error": self.simulation_error_message}

    def reset_simulation(self, start_datetime: Optional[datetime.datetime] = None):
        with self.lock:
            self.logger.info(f"Resetting simulation. Requested start: {start_datetime}")
            self._init_simulator(start_datetime)
            self.is_running = False;
            self.current_control_mode = "off"
            self.set_active_controller();
            self.manual_actuator_overrides = ControlState()  # Renamed
            self.simulation_error_message = None;
            self.logger.info("Simulation reset complete and paused.")
        simulation_control_event.set()

    def start_simulation(self, start_datetime_payload: Optional[Dict[str, Any]] = None):
        with self.lock:
            start_dt_obj: Optional[datetime.datetime] = None
            if start_datetime_payload:
                try:
                    s_year = start_datetime_payload.get("start_year", DEFAULT_START_YEAR);
                    s_month = start_datetime_payload.get("start_month", DEFAULT_START_MONTH)
                    s_day = start_datetime_payload.get("start_day", DEFAULT_START_DAY);
                    s_hour = start_datetime_payload.get("start_hour", DEFAULT_START_HOUR)
                    start_dt_obj = datetime.datetime(s_year, s_month, s_day, s_hour)
                    self.logger.info(f"Parsed start_datetime for start: {start_dt_obj}")
                    self._init_simulator(start_dt_obj)
                except (TypeError, ValueError) as e:
                    self.logger.error(f"Invalid start_datetime payload: {e}. Using current/default.");
                    if not self.simulator: self._init_simulator()
            elif not self.simulator:
                self._init_simulator()
            if self.simulator:
                self.is_running = True;
                self.simulation_error_message = None
                self.logger.info(
                    f"Sim started. Mode: {self.current_control_mode}. Time: {self.simulator.current_datetime.isoformat()}")
            else:
                self.logger.error("Cannot start: Simulator not initialized.")
        simulation_control_event.set()

    def stop_simulation(self):  
        with self.lock: self.is_running = False
        simulation_control_event.set();
        self.logger.info("Simulation stopped.")

    def set_control_mode(self, mode: str) -> bool:  
        with self.lock:
            try:
                mode_lower = mode.lower()
                if mode_lower not in VALID_CONTROL_MODES: 
                    self.logger.warning(f"Invalid control mode: {mode_lower}")
                    return False
                
                # Store previous state for rollback if needed
                previous_mode = self.current_control_mode
                previous_controller = self.active_controller
                
                # Update control mode
                self.current_control_mode = mode_lower
                
                # Try to set the new controller
                try:
                    self.set_active_controller()
                except Exception as e:
                    # Rollback on failure
                    self.current_control_mode = previous_mode
                    self.active_controller = previous_controller
                    self.logger.error(f"Failed to set controller for mode {mode_lower}: {e}")
                    return False
                
                self.logger.info(f"Control mode set to: {self.current_control_mode}")
                simulation_control_event.set()
                return True
            
            except Exception as e:
                self.logger.error(f"Error in set_control_mode: {e}", exc_info=True)
                return False

    def set_manual_overrides(self, overrides: ControlState): 
        with self.lock:
            self.manual_actuator_overrides = overrides
            if self.current_control_mode != "manual":
                self.logger.info("Switching to manual control mode due to override set.");
                self.set_control_mode("manual")
            else:
                self.logger.info(f"Manual overrides updated: {overrides.to_dict()}")
            simulation_control_event.set()

    def set_city_and_reload(self, new_city: str, initial_load: bool = False):
        with self.lock:
            new_city_lower = new_city.lower()
            if new_city_lower not in VALID_CITIES:
                self.logger.error(f"Invalid city: {new_city_lower}");
                if not initial_load: raise ValueError(f"Invalid city: {new_city_lower}"); return
            if not initial_load and new_city_lower == self.city:
                self.logger.info(f"City {new_city_lower} already set.");
                return
            self.logger.info(f"Setting city to {new_city_lower} (Initial: {initial_load})")
            self.is_running = False
            self.city = new_city_lower
            try:
                self._load_city_config();
                self._load_all_controllers_for_city()  # Renamed
                self._init_simulator()
                if self.simulator:
                    self.current_control_mode = "off";
                    self.set_active_controller()  # Renamed
                    self.manual_actuator_overrides = ControlState();
                    self.simulation_error_message = None
                    self.logger.info(f"City set to {self.city}. Sim reset. Mode 'off'.")
                else:
                    self.logger.error(f"Sim failed init after city change to {self.city}.")
                    self.simulation_error_message = f"Failed to init sim for city {self.city}."
            except Exception as e:
                self.logger.error(f"Failed to set city/reload for {new_city_lower}: {e}", exc_info=True)
                self.simulation_error_message = f"Failed to set city: {e}"
                if not initial_load: raise
            finally:
                if not initial_load: simulation_control_event.set()

    def set_user_simulator_params(self, params: Dict[str, float]): 
        with self.lock:
            self.logger.info(f"Updating user simulator params with: {params}")
            changed_any = False
            for key, value in params.items():
                if key in self.user_simulator_params:
                    try:
                        new_val = float(value)
                        if self.user_simulator_params[key] != new_val:
                            self.user_simulator_params[key] = new_val;
                            changed_any = True
                            self.logger.info(f"User sim param '{key}' updated to: {new_val}")
                    except ValueError:
                        self.logger.warning(f"Invalid value for {key}: {value}. Must be float.")
                else:
                    self.logger.warning(f"Unknown sim param: {key}")
            if changed_any:
                self.logger.info(f"User sim params now: {self.user_simulator_params}")
                self.simulation_error_message = "Sim params updated. Reset simulation for changes to take full effect."
                if self.is_running: self.is_running = False; self.logger.info(
                    "Sim paused due to param change. Please reset.")
            else:
                self.logger.info("No changes to user sim params (values same or keys invalid).")
        simulation_control_event.set()

    def run_batch_simulation(self, duration_hours: int, target_control_mode: str) -> Dict[str, Any]:
        with self.lock:
            if self.is_running:
                self.logger.warning("Cannot start batch run: Live simulation is running.")
                return {"status": "error", "detail": "Live simulation is running. Stop it before starting a batch run."}

            self.logger.info(
                f"Starting batch simulation: {duration_hours} hrs, Mode: {target_control_mode}, City: {self.city}")
            batch_sim_params = self.user_simulator_params.copy()

            # Use the simulator's current time if available and paused, otherwise default.
            # This ensures if user sets a specific date via reset, batch starts from there.
            if self.simulator and not self.is_running:
                start_dt_batch = self.simulator.current_datetime
            else:  # Default if no simulator or if live sim was running (though we block this case above)
                start_dt_batch = datetime.datetime(DEFAULT_START_YEAR, DEFAULT_START_MONTH, DEFAULT_START_DAY,
                                                   DEFAULT_START_HOUR)

            try:
                batch_simulator = GreenhouseSimulator(
                    city_config_path=self.city_config_path, init_state=None,
                    start_year=start_dt_batch.year, start_month=start_dt_batch.month,
                    start_day_of_month=start_dt_batch.day, start_hour=start_dt_batch.hour,
                    **batch_sim_params)
            except Exception as e:
                self.logger.error(f"Failed to initialize batch simulator: {e}", exc_info=True)
                return {"status": "error", "detail": f"Failed to initialize batch simulator: {e}"}

            batch_controller: Any = None
            # Load reward_params specific to the ML agent being batch-tested
            batch_reward_params: Dict[str, float] = {}

            if target_control_mode == "ml_normal":
                batch_controller = self.agents.get(f"{self.city}_normal")
                if batch_controller: batch_reward_params = self._load_reward_params_for_agent("normal")
            elif target_control_mode == "ml_eco":
                batch_controller = self.agents.get(f"{self.city}_eco")
                if batch_controller: batch_reward_params = self._load_reward_params_for_agent("eco")
            elif target_control_mode == "baseline_normal":
                if not self.city_config_data: self._load_city_config()  # Ensure city_config is loaded
                batch_controller = BaselineController(city_config=self.city_config_data,
                                                      target_ranges=self.ml_training_target_ranges, eco_mode=False)
            elif target_control_mode == "baseline_eco":
                if not self.city_config_data: self._load_city_config()
                batch_controller = BaselineController(city_config=self.city_config_data,
                                                      target_ranges=self.ml_training_target_ranges, eco_mode=True)
            else:
                return {"status": "error", "detail": f"Unsupported control mode for batch run: {target_control_mode}"}

            if batch_controller is None or (
                    isinstance(batch_controller, SmartMLAgent) and batch_controller.model is None):
                return {"status": "error",
                        "detail": f"Controller for {target_control_mode} not available for batch run."}

            num_steps = int(duration_hours * 60 / SIM_DT_MIN)
            batch_step_details = []

            # Initialize state for batch run
            b_outside_temp = batch_simulator.get_outside_temp();
            b_outside_rh = batch_simulator.get_outside_rh()
            current_batch_snapshot = SensorSnapshot(
                batch_simulator.temp_c, batch_simulator.rh, batch_simulator.sm,
                vapor_pressure_deficit(batch_simulator.temp_c, batch_simulator.rh),
                batch_simulator.rain_tank, batch_simulator.current_datetime.isoformat(),
                outside_temp=b_outside_temp, outside_rh=b_outside_rh)

            batch_actuator_history_for_ml = collections.deque([ControlState().to_dict()] * N_ACTUATOR_HISTORY_STEPS,
                                                              maxlen=N_ACTUATOR_HISTORY_STEPS)
            previous_requested_controls_batch = ControlState()

            # Counters for reward diagnostics within the batch run
            temp_out_of_band_steps_batch = 0
            rh_out_of_band_steps_batch = 0
            sm_out_of_band_steps_batch = 0
            all_vars_in_band_counter_batch = 0

            self.logger.info(f"Running batch simulation for {num_steps} steps...")
            for step_num in range(num_steps):
                prev_batch_snap_for_reward = current_batch_snapshot
                prev_batch_res_for_reward = batch_simulator.resource_totals().copy()
                requested_controls_batch: ControlState

                if isinstance(batch_controller, SmartMLAgent):
                    b_hour = batch_simulator.current_datetime.hour
                    b_base_obs_dict = {
                        't_c': current_batch_snapshot.t_c, 'rh': current_batch_snapshot.rh,
                        'sm': current_batch_snapshot.sm,
                        'vpd': current_batch_snapshot.vpd, 'rain_tank_l': current_batch_snapshot.rain_tank_l,
                        'outside_temp': current_batch_snapshot.outside_temp,
                        'outside_rh': current_batch_snapshot.outside_rh,
                        'sin_hour': math.sin(2 * math.pi * b_hour / 24.0),
                        'cos_hour': math.cos(2 * math.pi * b_hour / 24.0),
                        'is_raining': float(batch_simulator._is_currently_raining()),
                        **{f"target_{k}": v for k, v in self.ml_training_target_ranges.items()}}
                    try:
                        b_base_vals = [b_base_obs_dict[f] for f in OBSERVATION_FEATURE_ORDER[:BASE_OBS_DIM]]
                        b_hist_vals = [];
                        b_temp_hist = list(batch_actuator_history_for_ml)
                        for i in range(1, N_ACTUATOR_HISTORY_STEPS + 1):
                            past_ctrl = b_temp_hist[-i];
                            for act_name in ACTUATORS: b_hist_vals.append(float(past_ctrl.get(act_name, False)))
                        b_comb_vals = b_base_vals + b_hist_vals
                        b_obs_np = np.array(b_comb_vals, dtype=np.float32)
                        requested_controls_batch = batch_controller.get_action(b_obs_np)
                    except Exception as e_ml:
                        self.logger.error(f"Batch ML error: {e_ml}"); requested_controls_batch = ControlState()
                elif isinstance(batch_controller, BaselineController):
                    requested_controls_batch = batch_controller.get_controls(current_batch_snapshot,
                                                                             batch_simulator.get_outside_temp(),
                                                                             batch_simulator.get_outside_rh(),
                                                                             batch_simulator._is_currently_raining())
                else:
                    requested_controls_batch = ControlState()

                # --- Store snapshot *before* step for reward calculation ---
                prev_batch_snap_for_reward_actual = SensorSnapshot(
                    batch_simulator.temp_c, batch_simulator.rh, batch_simulator.sm,
                    vapor_pressure_deficit(batch_simulator.temp_c, batch_simulator.rh),
                    batch_simulator.rain_tank, batch_simulator.current_datetime.isoformat(),
                    outside_temp=batch_simulator.get_outside_temp(), outside_rh=batch_simulator.get_outside_rh()
                )

                current_batch_snapshot = batch_simulator.step(SIM_DT_MIN, requested_controls_batch,
                                                              (self.ml_training_target_ranges["t_c_min"] +
                                                               self.ml_training_target_ranges["t_c_max"]) / 2.0)

                step_diagnostics = {}
                if isinstance(batch_controller, SmartMLAgent) and batch_reward_params:
                    cur_batch_res = batch_simulator.resource_totals()
                    delta_kwh_batch = cur_batch_res['kwh'] - prev_batch_res_for_reward['kwh']
                    delta_water_batch = cur_batch_res['water_l'] - prev_batch_res_for_reward['water_l']

                    step_diagnostics['base_reward'] = calculate_reward(
                        prev_batch_snap_for_reward_actual,  # Use state *before* action
                        requested_controls_batch,
                        current_batch_snapshot,  # State *after* action
                        self.ml_training_target_ranges,
                        {'kwh': delta_kwh_batch, 'water_l': delta_water_batch},
                        batch_reward_params
                    )

                    num_chgs_batch = sum(
                        [getattr(requested_controls_batch, act) != getattr(previous_requested_controls_batch, act) for
                         act in ACTUATORS])
                    step_diagnostics['action_change_penalty'] = -batch_reward_params.get('W_ACTION_CHANGE',
                                                                                         0.05) * num_chgs_batch

                    # Replicate rule violation penalty (simplified from GreenhouseEnv)
                    rule_violations = 0
                    if requested_controls_batch.irrigation and (
                            prev_batch_snap_for_reward_actual.sm > self.ml_training_target_ranges[
                        'sm_max'] or prev_batch_snap_for_reward_actual.rain_tank_l < 1.0): rule_violations += 1
                    if requested_controls_batch.ac and requested_controls_batch.vent: rule_violations += 1
                    temp_buf_rule = batch_reward_params.get('TEMP_BUFFER', 0.5)
                    tmin_strict_rule = self.ml_training_target_ranges['t_c_min'] + temp_buf_rule
                    tmax_strict_rule = self.ml_training_target_ranges['t_c_max'] - temp_buf_rule
                    if requested_controls_batch.ac and (
                            tmin_strict_rule <= prev_batch_snap_for_reward_actual.t_c <= tmax_strict_rule): rule_violations += 1
                    step_diagnostics['rule_penalty'] = -float(
                        rule_violations * batch_reward_params.get('W_RULE_BREAK', 2.5))

                    # Replicate Stability & Sustained logic
                    temp_ok_batch = self.ml_training_target_ranges['t_c_min'] <= current_batch_snapshot.t_c <= \
                                    self.ml_training_target_ranges['t_c_max']
                    rh_ok_batch = self.ml_training_target_ranges['rh_min'] <= current_batch_snapshot.rh <= \
                                  self.ml_training_target_ranges['rh_max']
                    sm_ok_batch = self.ml_training_target_ranges['sm_min'] <= current_batch_snapshot.sm <= \
                                  self.ml_training_target_ranges['sm_max']

                    if not temp_ok_batch:
                        temp_out_of_band_steps_batch += 1
                    else:
                        temp_out_of_band_steps_batch = 0
                    step_diagnostics['sustained_t_penalty'] = -batch_reward_params.get('W_SUSTAINED_T_OUT',
                                                                                       5.0) if temp_out_of_band_steps_batch > (
                                batch_reward_params.get("N_T_SUSTAINED_HOURS", 1.0) * 60 / SIM_DT_MIN) else 0.0

                    if not rh_ok_batch:
                        rh_out_of_band_steps_batch += 1
                    else:
                        rh_out_of_band_steps_batch = 0
                    step_diagnostics['sustained_rh_penalty'] = -batch_reward_params.get('W_SUSTAINED_RH_OUT',
                                                                                        3.0) if rh_out_of_band_steps_batch > (
                                batch_reward_params.get("N_RH_SUSTAINED_HOURS", 1.0) * 60 / SIM_DT_MIN) else 0.0

                    if not sm_ok_batch:
                        sm_out_of_band_steps_batch += 1
                    else:
                        sm_out_of_band_steps_batch = 0
                    step_diagnostics['sustained_sm_penalty'] = -batch_reward_params.get('W_SUSTAINED_SM_OUT',
                                                                                        4.0) if sm_out_of_band_steps_batch > (
                                batch_reward_params.get("N_SM_SUSTAINED_HOURS", 3.0) * 60 / SIM_DT_MIN) else 0.0

                    if temp_ok_batch and rh_ok_batch and sm_ok_batch:
                        all_vars_in_band_counter_batch += 1
                    else:
                        all_vars_in_band_counter_batch = 0
                    step_diagnostics['stability_bonus'] = batch_reward_params.get('W_STABILITY_BONUS',
                                                                                  10.0) if all_vars_in_band_counter_batch > (
                                batch_reward_params.get("N_STABILITY_HOURS", 2.0) * 60 / SIM_DT_MIN) else 0.0

                    sensible_action_term_batch = 0.0
                    if requested_controls_batch.ac and (prev_batch_snap_for_reward_actual.t_c < (
                            self.ml_training_target_ranges['t_c_min'] - batch_reward_params.get('TEMP_BUFFER',
                                                                                                0.5) - 1.0)):
                        sensible_action_term_batch -= batch_reward_params.get('W_AC_WHEN_COLD_PENALTY', 1.5)
                    if prev_batch_snap_for_reward_actual.rh > (self.ml_training_target_ranges['rh_max'] + 10.0) and \
                            batch_simulator.get_outside_rh() < (prev_batch_snap_for_reward_actual.rh - 10.0) and \
                            not requested_controls_batch.vent:
                        sensible_action_term_batch -= batch_reward_params.get('W_NO_VENT_WHEN_HUMID_DRY_OUTSIDE', 0.75)
                    if prev_batch_snap_for_reward_actual.sm > (
                            self.ml_training_target_ranges['sm_max'] - batch_reward_params.get('SM_BUFFER', 3.0)) and \
                            not requested_controls_batch.irrigation:
                        sensible_action_term_batch += batch_reward_params.get('W_SENSIBLE_IRR_HOLD_BONUS', 0.3)
                    step_diagnostics['sensible_action_term'] = sensible_action_term_batch

                batch_step_details.append({
                    "step": step_num, "time_iso": current_batch_snapshot.current_datetime_iso,
                    "t_c": current_batch_snapshot.t_c, "rh": current_batch_snapshot.rh, "sm": current_batch_snapshot.sm,
                    "outside_temp": current_batch_snapshot.outside_temp,
                    "outside_rh": current_batch_snapshot.outside_rh,
                    "requested_controls": requested_controls_batch.to_dict(),
                    "applied_controls": batch_simulator.get_actual_actuator_states().to_dict(),
                    "diagnostics": step_diagnostics})
                batch_actuator_history_for_ml.append(batch_simulator.get_actual_actuator_states().to_dict())
                previous_requested_controls_batch = requested_controls_batch

            final_resources = batch_simulator.resource_totals()
            results = {
                "status": "Batch simulation completed", "city": self.city,
                "control_mode_used": target_control_mode, "duration_simulated_hours": duration_hours,
                "final_state": current_batch_snapshot.to_dict(), "final_resources": final_resources,
                "user_simulator_params_used": batch_sim_params, "step_details": batch_step_details}
            self.logger.info(f"Batch simulation finished. Final Resources: {results['final_resources']}")
            return results

# --- Global Instance & Background Thread ---
simulation_control_event = threading.Event()
manager: Optional[SimulationManager] = None
try:
    manager = SimulationManager()
except Exception as e:
    logger.critical(f"FATAL: Global SimulationManager failed to initialize: {e}", exc_info=True)

# --- Background Task ---
def _run_loop():
    global manager
    while True:
        if manager and manager.is_running:
            try:
                manager.step()
            except Exception as e:
                manager.logger.error(f"Error in simulation step: {e}", exc_info=True)
                manager.simulation_error_message = str(e)
        time.sleep(REAL_TIME_STEP_SEC)

# --- Startup Event ---
@app.on_event("startup")
async def on_startup():
    global manager
    manager = SimulationManager()
    threading.Thread(target=_run_loop, daemon=True).start()

# --- Manager Dependency ---
def get_manager_dependency() -> SimulationManager:
    if manager is None or manager.simulator is None:
        raise ResourceError("Simulation manager or core simulator not available.")
    return manager

# --- Serve Dashboard ---
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.isfile(index_path):
        return HTMLResponse(
            "<html><body><h1>Error: index.html not found</h1></body></html>",
            status_code=404
        )
    with open(index_path) as f:
        return HTMLResponse(f.read())

if __name__ == "__main__":
    import uvicorn

    logger.info("Running Uvicorn server directly from app.py script...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

