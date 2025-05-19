import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import os
import math
import random
import traceback
import collections
import datetime
from typing import Dict, Any, Optional, Tuple
import logging

# --- Basic Logging Setup ---
# Ensures logger is configured, especially important for subprocesses
if not logging.getLogger().hasHandlers() or logging.getLogger().getEffectiveLevel() > logging.INFO:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s')
logger_module = logging.getLogger(__name__)  # Logger for this module's messages

# --- Project Path Setup ---
try:
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except Exception as e:
    logger_module.warning(f"Could not modify sys.path for custom_env_wrapper. Error: {e}")

# --- Module Imports & Global Constant Definitions ---
# These need to be resolvable when this module is loaded by each subprocess.
try:
    # Import from simulator.core
    from simulator.core import (
        GreenhouseSimulator, SensorSnapshot, ControlState, vapor_pressure_deficit,
        TEMP_FLOOR as CORE_TEMP_FLOOR, TEMP_CAP as CORE_TEMP_CAP,
        RH_TERMINATE_MIN as CORE_RH_TERMINATE_MIN, RH_TERMINATE_MAX as CORE_RH_TERMINATE_MAX,
        SM_TERMINATE_MIN as CORE_SM_TERMINATE_MIN
    )
    # Import from ml_training.reward_functions
    from ml_training.reward_functions import calculate_reward
    # Import from evaluation.scenarios (single source of truth for observation structure)
    from evaluation.scenarios import (
        OBSERVATION_FEATURE_ORDER as _ENV_OBS_ORDER,  # Use _ to indicate they are for module setup
        N_ACTUATOR_HISTORY_STEPS as _ENV_N_HIST_STEPS,
        ACTUATORS as _ENV_ACTUATORS,
        BASE_OBSERVATION_FEATURE_ORDER as _ENV_BASE_OBS_ORDER,
        HISTORY_FEATURES_ORDERED as _ENV_HIST_FEATURES_ORDER  # Not used directly here but good for consistency
    )

    # --- Define module-level constants directly from imported values ---
    OBSERVATION_FEATURE_ORDER = _ENV_OBS_ORDER
    BASE_OBSERVATION_FEATURE_ORDER = _ENV_BASE_OBS_ORDER
    N_ACTUATOR_HISTORY_STEPS = _ENV_N_HIST_STEPS
    ACTUATORS = _ENV_ACTUATORS  # This is the critical list
    N_ACTUATORS = len(ACTUATORS)

    OBS_DIM = len(OBSERVATION_FEATURE_ORDER)
    BASE_OBS_DIM = len(BASE_OBSERVATION_FEATURE_ORDER)

    TEMP_FLOOR = CORE_TEMP_FLOOR
    TEMP_CAP = CORE_TEMP_CAP
    RH_TERMINATE_MIN = CORE_RH_TERMINATE_MIN
    RH_TERMINATE_MAX = CORE_RH_TERMINATE_MAX
    SM_TERMINATE_MIN = CORE_SM_TERMINATE_MIN

    expected_dim = BASE_OBS_DIM + (N_ACTUATORS * N_ACTUATOR_HISTORY_STEPS)
    if OBS_DIM != expected_dim:
        logger_module.critical(f"CRITICAL DIMENSION MISMATCH (custom_env): OBS_DIM={OBS_DIM}, expected={expected_dim}")
        raise ValueError("Observation dimension config error in custom_env_wrapper.py")
    logger_module.info("Successfully imported and defined constants for GreenhouseEnv.")

except ImportError as e:
    logger_module.critical(f"CRITICAL Import ERROR in custom_env_wrapper: {e}", exc_info=True)
    raise  # Stop execution if critical imports fail

# --- Observation Space Bounds ( match OBS_DIM=28) ---
OBS_LOW_BASE_ENV = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -20.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            dtype=np.float32)
OBS_HIGH_BASE_ENV = np.array(
    [50.0, 100.0, 100.0, 5.0, 500.0, 50.0, 100.0, 1.0, 1.0, 1.0, 50.0, 50.0, 100.0, 100.0, 100.0, 100.0],
    dtype=np.float32)
OBS_LOW_HIST_ENV = np.zeros(N_ACTUATORS * N_ACTUATOR_HISTORY_STEPS, dtype=np.float32)
OBS_HIGH_HIST_ENV = np.ones(N_ACTUATORS * N_ACTUATOR_HISTORY_STEPS, dtype=np.float32)

if OBS_LOW_BASE_ENV.shape[0] != BASE_OBS_DIM or OBS_LOW_HIST_ENV.shape[0] != (N_ACTUATORS * N_ACTUATOR_HISTORY_STEPS):
    logger_module.critical("Dimension mismatch in manual OBS_LOW/HIGH_BASE_ENV or HIST_ENV definitions.")
    raise ValueError("Observation space bounds definition error.")

OBS_LOW = np.concatenate((OBS_LOW_BASE_ENV, OBS_LOW_HIST_ENV))
OBS_HIGH = np.concatenate((OBS_HIGH_BASE_ENV, OBS_HIGH_HIST_ENV))

if OBS_DIM != OBS_LOW.shape[0]:
    logger_module.critical(f"Final OBS_DIM ({OBS_DIM}) != constructed OBS_LOW shape ({OBS_LOW.shape[0]}).")
    raise ValueError("Final observation space dimension configuration error.")

# --- Other Module-Level Constants ---
ACTION_DIM = 24
DEFAULT_DT_MIN = 5
DEFAULT_MAX_DAYS = 30
DEFAULT_START_YEAR = 2025;
DEFAULT_START_MONTH = 7;
DEFAULT_START_DAY = 1;
DEFAULT_START_HOUR = 6


class GreenhouseEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 10}

    def __init__(self,
                 city_config_path: str,
                 target_ranges: Dict[str, float],
                 mode: str,
                 reward_params: Dict[str, float],
                 dt_min: int = DEFAULT_DT_MIN,
                 max_days: int = DEFAULT_MAX_DAYS,
                 user_simulator_params: Optional[Dict[str, float]] = None
                 ):
        super().__init__()
        self.logger = logging.getLogger(f"GreenhouseEnv.{os.path.basename(city_config_path).split('.')[0]}.{mode}")

        if not os.path.exists(city_config_path):
            self.logger.error(f"City config not found: {city_config_path}")
            raise FileNotFoundError(f"City config not found: {city_config_path}")
        self.city_config_path = city_config_path
        self.target_ranges = target_ranges
        self.mode = mode
        self.reward_params = reward_params
        self.dt_min = dt_min
        self.max_steps = int(max_days * 24 * 60 / dt_min)
        self.user_simulator_params = user_simulator_params if user_simulator_params is not None else {}

        # These use the module-level constants defined above
        self.observation_space = spaces.Box(OBS_LOW, OBS_HIGH, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(ACTION_DIM)

        self.simulator: Optional[GreenhouseSimulator] = None
        self.current_step = 0
        self.last_sensor_snapshot: Optional[SensorSnapshot] = None
        self.last_resource_totals: Dict[str, float] = {'kwh': 0.0, 'water_l': 0.0}

        self.actuator_state_history_buffer: collections.deque[ControlState] = \
            collections.deque([ControlState()] * N_ACTUATOR_HISTORY_STEPS, maxlen=N_ACTUATOR_HISTORY_STEPS)
        self.previous_requested_controls: ControlState = ControlState()

        self.temp_out_of_band_steps = 0
        self.rh_out_of_band_steps = 0
        self.sm_out_of_band_steps = 0
        self.all_vars_in_band_counter = 0

        self.logger.info(f"Initialized. OBS_DIM={OBS_DIM}, max_steps={self.max_steps}")

    def _get_observation(self) -> np.ndarray:
        if self.simulator is None or self.last_sensor_snapshot is None:
            self.logger.warning("Simulator/snapshot not available for _get_observation. Returning zeros.")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        sim = self.simulator;
        snap = self.last_sensor_snapshot
        hr = sim.current_datetime.hour

        base_obs_dict = {
            't_c': snap.t_c, 'rh': snap.rh, 'sm': snap.sm, 'vpd': snap.vpd, 'rain_tank_l': snap.rain_tank_l,
            'outside_temp': sim.get_outside_temp(), 'outside_rh': sim.get_outside_rh(),
            'sin_hour': math.sin(2 * math.pi * hr / 24.0), 'cos_hour': math.cos(2 * math.pi * hr / 24.0),
            'is_raining': float(sim._is_currently_raining()),
            'target_t_c_min': self.target_ranges['t_c_min'], 'target_t_c_max': self.target_ranges['t_c_max'],
            'target_rh_min': self.target_ranges['rh_min'], 'target_rh_max': self.target_ranges['rh_max'],
            'target_sm_min': self.target_ranges['sm_min'], 'target_sm_max': self.target_ranges['sm_max'],
        }
        history_values = []
        # Iterate using module-level ACTUATORS
        for past_control_state in self.actuator_state_history_buffer:
            for act_name in ACTUATORS:
                history_values.append(float(getattr(past_control_state, act_name, False)))
        try:
            # Iterate using module-level BASE_OBSERVATION_FEATURE_ORDER
            base_values = [base_obs_dict[f] for f in BASE_OBSERVATION_FEATURE_ORDER]
            if len(history_values) != (N_ACTUATORS * N_ACTUATOR_HISTORY_STEPS):  # Use module-level constants
                raise ValueError(
                    f"History length error. Expected {N_ACTUATORS * N_ACTUATOR_HISTORY_STEPS}, got {len(history_values)}")
            combined_values = base_values + history_values
            arr = np.array(combined_values, dtype=np.float32)
            if arr.shape != (OBS_DIM,): raise ValueError(
                f"Final obs shape error. Expected ({OBS_DIM},), got {arr.shape}")  # Use module-level OBS_DIM
            return np.clip(arr, self.observation_space.low, self.observation_space.high)
        except KeyError as e:
            self.logger.error(f"Missing key '{e}' building obs array!", exc_info=True)
            return np.full(self.observation_space.shape, np.nan, dtype=np.float32)
        except ValueError as e:
            self.logger.error(f"Value error building obs: {e}", exc_info=True)
            return np.full(self.observation_space.shape, np.nan, dtype=np.float32)

    def _map_discrete_action_to_controlstate(self, da: int) -> ControlState:
        if not 0 <= da < ACTION_DIM: self.logger.warning(f"Invalid action {da}. Defaulting OFF."); da = 0
        l, p = divmod(da, 8);
        return ControlState(fan=bool(p & 4), ac=(l > 0), vent=bool(p & 2), irrigation=bool(p & 1))

    def _rule_violation_penalty(self, rq_cs: ControlState, ps: SensorSnapshot) -> float:
        v = 0;
        w = self.reward_params.get('W_RULE_BREAK', 2.5);
        tr = self.target_ranges
        if rq_cs.irrigation and (ps.sm > tr['sm_max'] or ps.rain_tank_l < 1.0): v += 1
        if rq_cs.ac and rq_cs.vent: v += 1
        tb = self.reward_params.get('TEMP_BUFFER', 0.5);
        tms = tr['t_c_min'] + tb;
        tMxs = tr['t_c_max'] - tb
        if rq_cs.ac and (ps.t_c < tms): v += 1  # Only penalize AC when temp is below buffered min
        return -float(v * w)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        start_month_sim = random.randint(1, 12)
        try:
            self.simulator = GreenhouseSimulator(
                self.city_config_path, init_state=None, start_month=start_month_sim,
                start_year=DEFAULT_START_YEAR, start_day_of_month=DEFAULT_START_DAY, start_hour=DEFAULT_START_HOUR,
                **self.user_simulator_params
            )
        except Exception as e:
            self.logger.error(f"Failed to init simulator in reset: {e}", exc_info=True)
            return np.zeros(self.observation_space.shape, dtype=np.float32), {"error": str(e)}

        self.current_step = 0
        self.last_resource_totals = {'kwh': 0.0, 'water_l': 0.0}
        initial_sim_controls = self.simulator.get_actual_actuator_states()
        self.actuator_state_history_buffer.clear()
        for _ in range(N_ACTUATOR_HISTORY_STEPS): self.actuator_state_history_buffer.append(initial_sim_controls)
        self.previous_requested_controls = initial_sim_controls

        sim_dt_iso = self.simulator.current_datetime.isoformat()
        self.last_sensor_snapshot = SensorSnapshot(
            self.simulator.temp_c, self.simulator.rh, self.simulator.sm,
            vapor_pressure_deficit(self.simulator.temp_c, self.simulator.rh),
            self.simulator.rain_tank, sim_dt_iso)

        self.temp_out_of_band_steps = 0;
        self.rh_out_of_band_steps = 0;
        self.sm_out_of_band_steps = 0
        self.all_vars_in_band_counter = 0
        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.simulator is None or self.last_sensor_snapshot is None:
            self.logger.error("Step called but simulator not initialized.")
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {"error": "Sim not init"}

        sim = self.simulator;
        prev_snap = self.last_sensor_snapshot;
        prev_res = self.last_resource_totals.copy()
        target_mid = (self.target_ranges['t_c_min'] + self.target_ranges['t_c_max']) / 2.0

        requested_controls = self._map_discrete_action_to_controlstate(action)
        rule_penalty = self._rule_violation_penalty(requested_controls, prev_snap)

        action_change_penalty = 0.0
        w_action_change = self.reward_params.get('W_ACTION_CHANGE', 0.05)
        if w_action_change > 0:
            # Use module-level ACTUATORS
            num_chgs = sum(
                [getattr(requested_controls, act) != getattr(self.previous_requested_controls, act) for act in
                 ACTUATORS])
            action_change_penalty = -w_action_change * num_chgs

        try:
            next_snap = sim.step(dt_min=self.dt_min, requested_controls=requested_controls,
                                 target_temp_midpoint=target_mid)
        except Exception as e:
            self.logger.error(f"Simulator step error: {e}", exc_info=True)
            return self._get_observation(), -200.0, True, False, {'error': str(e)}

        cur_res = sim.resource_totals();
        delta_kwh = cur_res['kwh'] - prev_res['kwh'];
        delta_water = cur_res['water_l'] - prev_res['water_l']
        self.last_resource_totals = cur_res

        base_reward = calculate_reward(prev_snap, requested_controls, next_snap, self.target_ranges,
                                       {'kwh': delta_kwh, 'water_l': delta_water}, self.reward_params)

        temp_ok = self.target_ranges['t_c_min'] <= next_snap.t_c <= self.target_ranges['t_c_max']
        rh_ok = self.target_ranges['rh_min'] <= next_snap.rh <= self.target_ranges['rh_max']
        sm_ok = self.target_ranges['sm_min'] <= next_snap.sm <= self.target_ranges['sm_max']

        sustained_t_penalty = 0.0
        if not temp_ok:
            self.temp_out_of_band_steps += 1
        else:
            self.temp_out_of_band_steps = 0
        if self.temp_out_of_band_steps > (self.reward_params.get("N_T_SUSTAINED_HOURS", 1.0) * 60 / self.dt_min):
            sustained_t_penalty = -self.reward_params.get('W_SUSTAINED_T_OUT', 5.0)

        sustained_rh_penalty = 0.0
        if not rh_ok:
            self.rh_out_of_band_steps += 1
        else:
            self.rh_out_of_band_steps = 0
        if self.rh_out_of_band_steps > (self.reward_params.get("N_RH_SUSTAINED_HOURS", 1.0) * 60 / self.dt_min):
            sustained_rh_penalty = -self.reward_params.get('W_SUSTAINED_RH_OUT', 3.0)

        sustained_sm_penalty = 0.0
        if not sm_ok:
            self.sm_out_of_band_steps += 1
        else:
            self.sm_out_of_band_steps = 0
        if self.sm_out_of_band_steps > (self.reward_params.get("N_SM_SUSTAINED_HOURS", 3.0) * 60 / self.dt_min):
            sustained_sm_penalty = -self.reward_params.get('W_SUSTAINED_SM_OUT', 4.0)

        stability_bonus = 0.0
        if temp_ok and rh_ok and sm_ok:
            self.all_vars_in_band_counter += 1
        else:
            self.all_vars_in_band_counter = 0
        if self.all_vars_in_band_counter > (self.reward_params.get("N_STABILITY_HOURS", 2.0) * 60 / self.dt_min):
            stability_bonus = self.reward_params.get('W_STABILITY_BONUS', 10.0)

        sensible_action_term = 0.0
        if requested_controls.ac and (
                prev_snap.t_c < (self.target_ranges['t_c_min'] - self.reward_params.get('TEMP_BUFFER', 0.5) - 1.0)):
            sensible_action_term -= self.reward_params.get('W_AC_WHEN_COLD_PENALTY', 1.5)
        if prev_snap.rh > (self.target_ranges['rh_max'] + 10.0) and \
                sim.get_outside_rh() < (prev_snap.rh - 10.0) and \
                not requested_controls.vent:
            sensible_action_term -= self.reward_params.get('W_NO_VENT_WHEN_HUMID_DRY_OUTSIDE', 0.75)
        if prev_snap.sm > (self.target_ranges['sm_max'] - self.reward_params.get('SM_BUFFER', 3.0)) and \
                not requested_controls.irrigation:
            sensible_action_term += self.reward_params.get('W_SENSIBLE_IRR_HOLD_BONUS', 0.3)

        reward = base_reward + rule_penalty + action_change_penalty + \
                 sustained_t_penalty + sustained_rh_penalty + sustained_sm_penalty + \
                 stability_bonus + sensible_action_term

        self.current_step += 1
        self.last_sensor_snapshot = next_snap
        actual_applied_controls = sim.get_actual_actuator_states()
        self.actuator_state_history_buffer.append(actual_applied_controls)
        self.previous_requested_controls = requested_controls

        #  Termination Logic using module-level constants
        terminated = False
        # Use module-level TEMP_FLOOR, TEMP_CAP
        if not (TEMP_FLOOR <= next_snap.t_c <= TEMP_CAP):
            terminated = True
            self.logger.debug(f"Terminating due to Temperature: {next_snap.t_c:.1f}")
        # Use module-level RH_TERMINATE_MIN, RH_TERMINATE_MAX
        elif not (RH_TERMINATE_MIN <= next_snap.rh <= RH_TERMINATE_MAX):
            terminated = True
            self.logger.debug(f"Terminating due to RH: {next_snap.rh:.1f}")
        # Use module-level SM_TERMINATE_MIN
        elif next_snap.sm <= SM_TERMINATE_MIN:  # Terminate if SM hits the absolute floor
            terminated = True
            self.logger.debug(f"Terminating due to SM: {next_snap.sm:.1f}")

        truncated = self.current_step >= self.max_steps

        obs = self._get_observation()
        info = {
            'resources_kwh': delta_kwh, 'resources_water_l': delta_water,
            'base_reward': base_reward, 'rule_penalty': rule_penalty,
            'action_change_penalty': action_change_penalty, 'stability_bonus': stability_bonus,
            'sustained_t_penalty': sustained_t_penalty, 'sustained_rh_penalty': sustained_rh_penalty,
            'sustained_sm_penalty': sustained_sm_penalty, 'sensible_action_term': sensible_action_term,
            'is_terminated': terminated, 'is_truncated': truncated
        }
        return obs, float(reward), terminated, truncated, info

    def render(self, mode="human"):
        if mode == "ansi" and self.last_sensor_snapshot:
            s = self.last_sensor_snapshot;
            dt_iso = s.current_datetime_iso
            return f"Step {self.current_step}/{self.max_steps} | {dt_iso} | T={s.t_c:.1f}C RH={s.rh:.1f}% SM={s.sm:.1f}%"
        elif mode == "human":
            print(self.render("ansi"))

    def close(self):
        self.simulator = None
        self.logger.info("GreenhouseEnv closed.")