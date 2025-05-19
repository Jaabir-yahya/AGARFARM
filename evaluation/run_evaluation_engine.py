import pandas as pd
import yaml
import os
import time
import copy
import numpy as np
import math
import logging
import collections
from typing import Dict, Optional, Tuple, Any
import sys

# --- Project Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Imports from project ---
try:
    from simulator.core import (
        GreenhouseSimulator, SensorSnapshot, ControlState, vapor_pressure_deficit,
        TEMP_FLOOR, TEMP_CAP  # These are used by GreenhouseEnv, not directly here usually
    )
    from controllers.baseline import BaselineController
    from controllers.smart_ml_agent import SmartMLAgent
    from evaluation.scenarios import (
        get_scenario, EVAL_DT_MIN,
        OBSERVATION_FEATURE_ORDER as EVAL_OBS_ORDER,
        N_ACTUATOR_HISTORY_STEPS as EVAL_N_HIST_STEPS,
        ACTUATORS as EVAL_ACTUATORS,
        BASE_OBSERVATION_FEATURE_ORDER as EVAL_BASE_OBS_ORDER
    )
    from evaluation.metrics import calculate_metrics
    from evaluation.plot_utils import create_comparison_figure

    # Define module-level constants from imported values
    OBSERVATION_FEATURE_ORDER = EVAL_OBS_ORDER
    BASE_OBSERVATION_FEATURE_ORDER = EVAL_BASE_OBS_ORDER
    N_ACTUATOR_HISTORY_STEPS = EVAL_N_HIST_STEPS
    ACTUATORS = EVAL_ACTUATORS
    N_ACTUATORS = len(ACTUATORS)
    BASE_OBS_DIM = len(BASE_OBSERVATION_FEATURE_ORDER)

    # --- !!! DEFINE OBS_DIM AT MODULE LEVEL !!! ---
    OBS_DIM = len(OBSERVATION_FEATURE_ORDER)

    EXPECTED_OBS_DIM_EVAL = BASE_OBS_DIM + (N_ACTUATORS * N_ACTUATOR_HISTORY_STEPS)
    if OBS_DIM != EXPECTED_OBS_DIM_EVAL:
        logging.warning(
            f"Imported OBSERVATION_FEATURE_ORDER length results in OBS_DIM={OBS_DIM}, "
            f"calculated expected is {EXPECTED_OBS_DIM_EVAL}! Check scenarios.py definitions."
        )
    logging.info("Successfully imported modules for evaluation engine.")

except ImportError as e:
    logging.error(f"ERROR: Failed to import modules in run_evaluation_engine.py: {e}", exc_info=True)
    raise
except NameError as e:
    logging.error(f"ERROR: A required name (likely from scenarios.py) is not defined: {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s')

# --- Constants & Config ---
CITY_CONFIG_DIR = os.path.join(project_root, "simulator", "city_configs")
MODEL_DIR = os.path.join(project_root, "evaluation", "trained_models")
FIGURE_DIR = os.path.join(project_root, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

MAX_SIM_TIME_SEC_PER_RUN = 180
STABLE_EXIT_PCT = 0.95
STABLE_CHECK_START_PCT = 0.5
DEFAULT_EVAL_YEAR = 2025
DEFAULT_EVAL_DAY = 1
DEFAULT_EVAL_HOUR = 0


def run_single_simulation(
        city_name: str,
        scenario_id: str,
        controller_type: str,
        controller_display_name: str,
        model_suffix: Optional[str] = None,
        dt_min_override: Optional[int] = None
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    logger.info(
        f"  Starting single run: City={city_name}, Scenario={scenario_id}, Controller='{controller_display_name}'")
    start_run_time = time.time()
    history = []
    metrics_dict: Dict[str, Any] = {"error": "Run not completed"}

    try:
        scenario_config = get_scenario(scenario_id)
        city_config_path = os.path.join(CITY_CONFIG_DIR, f"{city_name}.yaml")
        if not os.path.exists(city_config_path):
            raise FileNotFoundError(f"City config not found: {city_config_path}")
        with open(city_config_path, 'r') as f:
            city_config = yaml.safe_load(f)

        target_ranges = scenario_config.get('target_ranges', {})
        if not target_ranges:
            target_ranges = {k.replace('default_target_', ''): v for k, v in city_config.items() if
                             k.startswith('default_target_')}
        for key_base in ['t_c', 'rh', 'sm']:
            if f'{key_base}_min' not in target_ranges or f'{key_base}_max' not in target_ranges:
                raise ValueError(f"Missing target range keys for {key_base} in {scenario_id}/{city_name}")

        sim_month_for_run = scenario_config.get('sim_month', city_config.get('default_sim_month', 7))
        start_eval_year = scenario_config.get('start_year', DEFAULT_EVAL_YEAR)
        start_eval_day = scenario_config.get('start_day', DEFAULT_EVAL_DAY)
        start_eval_hour = scenario_config.get('start_hour', DEFAULT_EVAL_HOUR)
        initial_state_override = scenario_config.get('initial_state', None)
        user_sim_params_eval = scenario_config.get('user_simulator_params',
                                                   {})  # For potential scenario-specific multipliers

        sim = GreenhouseSimulator(
            city_config_path=city_config_path, init_state=initial_state_override,
            start_year=start_eval_year, start_month=sim_month_for_run,
            start_day_of_month=start_eval_day, start_hour=start_eval_hour,
            **user_sim_params_eval
        )

        target_temp_midpoint = (target_ranges['t_c_min'] + target_ranges['t_c_max']) / 2.0
        current_dt_min = dt_min_override if dt_min_override is not None else EVAL_DT_MIN

        controller_instance: Any = None
        if controller_type == 'baseline':
            is_eco = "eco" in controller_display_name.lower()
            controller_instance = BaselineController(city_config=city_config, target_ranges=target_ranges,
                                                     eco_mode=is_eco)
            logger.info(f"Using Baseline Controller (Eco: {is_eco}) for '{controller_display_name}'")
        elif controller_type.startswith('ml_'):
            if not model_suffix: raise ValueError("model_suffix required for ML controllers")
            model_filepath = os.path.join(MODEL_DIR, f"ppo_agent_{model_suffix}.zip")
            vec_norm_filepath = os.path.join(MODEL_DIR, f"vecnormalize_{model_suffix}.pkl")
            vec_norm_path_arg = vec_norm_filepath if os.path.exists(vec_norm_filepath) else None
            controller_instance = SmartMLAgent(model_path=model_filepath, vec_normalize_path=vec_norm_path_arg)
            if controller_instance.model is None:
                raise RuntimeError(f"ML model failed to load: {model_filepath} for '{controller_display_name}'")

            expected_ml_obs_shape = (OBS_DIM,)  # Use module-level OBS_DIM
            if hasattr(controller_instance.model, 'observation_space') and \
                    controller_instance.model.observation_space.shape != expected_ml_obs_shape:
                logger.warning(
                    f"Loaded ML model {model_suffix} obs shape {controller_instance.model.observation_space.shape} != expected {expected_ml_obs_shape}.")
            logger.info(f"Using ML Controller: {os.path.basename(model_filepath)} for '{controller_display_name}'")
        else:
            raise ValueError(f"Unknown controller_type: {controller_type}")

        sim_dt_iso = sim.current_datetime.isoformat()
        # Pass current outside conditions to the initial snapshot
        current_outside_temp = sim.get_outside_temp()
        current_outside_rh = sim.get_outside_rh()
        snapshot = SensorSnapshot(sim.temp_c, sim.rh, sim.sm, vapor_pressure_deficit(sim.temp_c, sim.rh),
                                  sim.rain_tank, sim_dt_iso,
                                  outside_temp=current_outside_temp, outside_rh=current_outside_rh)

        total_ticks = scenario_config['duration_steps']
        stable_check_start_tick = int(total_ticks * STABLE_CHECK_START_PCT);
        ticks_in_band_target = int(total_ticks * STABLE_EXIT_PCT)
        consecutive_ticks_in_band = 0
        is_maintenance = scenario_config.get('is_maintenance', False);
        override_start = scenario_config.get('override_start_step', -1)
        override_end = override_start + scenario_config.get('override_duration_steps', 0)
        override_actions_dict = scenario_config.get('override_actuator_states', {})
        fixed_override_control = ControlState(
            **override_actions_dict) if is_maintenance and override_actions_dict else None

        initial_actual_controls = sim.get_actual_actuator_states()
        actuator_history: collections.deque[Dict[str, bool]] = collections.deque(
            [initial_actual_controls.to_dict()] * N_ACTUATOR_HISTORY_STEPS, maxlen=N_ACTUATOR_HISTORY_STEPS)

        logger.info(f"  Simulating '{controller_display_name}' for {total_ticks} ticks...")
        loop_start_time = time.time();
        aborted_early = False

        for tick in range(total_ticks):
            if time.time() - loop_start_time > MAX_SIM_TIME_SEC_PER_RUN:
                logger.warning(f"Run exceeded time limit. Aborting.");
                metrics_dict = {"error": f"Timeout at tick {tick}"};
                aborted_early = True;
                break

            current_sim_time = sim.current_datetime;
            outside_temp = sim.get_outside_temp()
            outside_rh = sim.get_outside_rh();
            current_hour = current_sim_time.hour
            is_raining_flag = sim._is_currently_raining()
            requested_controls: ControlState;
            run_override = False

            if is_maintenance and fixed_override_control and override_start <= tick < override_end:
                requested_controls = fixed_override_control;
                run_override = True
            elif isinstance(controller_instance, SmartMLAgent):
                base_obs_dict = {
                    't_c': snapshot.t_c, 'rh': snapshot.rh, 'sm': snapshot.sm, 'vpd': snapshot.vpd,
                    'rain_tank_l': snapshot.rain_tank_l,
                    'outside_temp': outside_temp, 'outside_rh': outside_rh,  # Use current outside conditions for obs
                    'sin_hour': math.sin(2 * math.pi * current_hour / 24.0),
                    'cos_hour': math.cos(2 * math.pi * current_hour / 24.0),
                    'is_raining': float(is_raining_flag),
                    **{f"target_{k}": v for k, v in target_ranges.items()}}

                base_values = [base_obs_dict[f] for f in OBSERVATION_FEATURE_ORDER[:BASE_OBS_DIM]]
                history_values = []
                temp_history_list = list(actuator_history)
                for i in range(1, N_ACTUATOR_HISTORY_STEPS + 1):
                    past_control_dict = temp_history_list[-i]
                    for act_name_hist in ACTUATORS:
                        history_values.append(float(past_control_dict.get(act_name_hist, False)))

                if len(history_values) != (N_ACTUATORS * N_ACTUATOR_HISTORY_STEPS):
                    logger.error(
                        f"Eval history length mismatch: got {len(history_values)}, expected {N_ACTUATORS * N_ACTUATOR_HISTORY_STEPS}")
                    history_values = [0.0] * (N_ACTUATORS * N_ACTUATOR_HISTORY_STEPS)

                combined_values = base_values + history_values
                observation_np = np.array(combined_values, dtype=np.float32)

                if observation_np.shape != (OBS_DIM,):
                    logger.error(
                        f"Eval observation shape {observation_np.shape} != ({OBS_DIM},). Using default controls.")
                    requested_controls = ControlState()
                else:
                    requested_controls = controller_instance.get_action(observation_np)
            elif isinstance(controller_instance, BaselineController):
                requested_controls = controller_instance.get_controls(snapshot, outside_temp, outside_rh,
                                                                      is_raining_flag)
            else:
                requested_controls = ControlState()

            current_resources = sim.resource_totals()
            log_entry = {
                "tick": tick, "city": city_name, "scenario": scenario_id, "controller_name": controller_display_name,
                "t_c": snapshot.t_c, "rh": snapshot.rh, "sm": snapshot.sm, "vpd": snapshot.vpd,
                "rain_tank_l": snapshot.rain_tank_l, "current_datetime_iso": snapshot.current_datetime_iso,
                "outside_temp": outside_temp, "outside_rh": outside_rh,  # Log current outside conditions
                "fan": int(requested_controls.fan), "ac": int(requested_controls.ac),
                "vent": int(requested_controls.vent), "irrigation": int(requested_controls.irrigation),
                "kwh": current_resources["kwh"], "water_l": current_resources["water_l"],
                "is_raining": int(is_raining_flag), "override_active": int(run_override)}
            history.append(log_entry)
            actuator_history.append(requested_controls.to_dict())
            snapshot = sim.step(dt_min=current_dt_min, requested_controls=requested_controls,
                                target_temp_midpoint=target_temp_midpoint)

            if tick > stable_check_start_tick:
                temp_in = target_ranges['t_c_min'] <= snapshot.t_c <= target_ranges['t_c_max']
                rh_in = target_ranges['rh_min'] <= snapshot.rh <= target_ranges['rh_max']
                sm_in = target_ranges['sm_min'] <= snapshot.sm <= target_ranges['sm_max']
                if temp_in and rh_in and sm_in:
                    consecutive_ticks_in_band += 1
                else:
                    consecutive_ticks_in_band = 0
                if consecutive_ticks_in_band >= ticks_in_band_target:
                    logger.info(f"Stable in target bands, exiting early at tick {tick}.");
                    aborted_early = True;
                    break

        final_tick_value = total_ticks if not aborted_early else tick + 1
        current_resources = sim.resource_totals()
        final_log_entry = {
            "tick": final_tick_value, "city": city_name, "scenario": scenario_id,
            "controller_name": controller_display_name,
            **snapshot.to_dict(),  # This will now include outside_temp/rh if SensorSnapshot was updated
            # "outside_temp": sim.get_outside_temp(), "outside_rh": sim.get_outside_rh(), # Redundant if in snapshot.to_dict()
            "fan": np.nan, "ac": np.nan, "vent": np.nan, "irrigation": np.nan,
            "kwh": current_resources["kwh"], "water_l": current_resources["water_l"],
            "is_raining": int(sim._is_currently_raining()), "override_active": np.nan}
        history.append(final_log_entry)

        history_df = pd.DataFrame(history) if history else pd.DataFrame()
        if metrics_dict.get("error") == "Run not completed": metrics_dict = {}

        if not ("Time limit exceeded" in metrics_dict.get("error", "") or "ERROR" in str(
                metrics_dict.get("error", "")).upper()):
            calculated_metrics_data = calculate_metrics(history_df, target_ranges)
            metrics_dict.update(calculated_metrics_data)
        elif "Time limit exceeded" not in metrics_dict.get("error", ""):
            metrics_dict["error"] = metrics_dict.get("error", "Aborted early or other error")

        end_run_time = time.time()
        logger.info(
            f"    -> Finished run for '{controller_display_name}'. Duration: {end_run_time - start_run_time:.2f}s.")
        return history_df, metrics_dict

    except Exception as e:
        logger.error(f"CRITICAL ERROR in run_single_simulation for '{controller_display_name}': {e}", exc_info=True)
        history_df_on_error = pd.DataFrame(history) if history else pd.DataFrame()
        error_msg = metrics_dict.get("error") if metrics_dict and metrics_dict.get(
            "error") != "Run not completed" else str(e)
        return history_df_on_error, {"error": error_msg}


def run_full_comparison(city_name: str, scenario_id: str) -> Dict[str, Any]:
    logger.info(f"--- Running Full Comparison for {city_name} / {scenario_id} ---")
    results: Dict[str, Any] = {}
    history_data: Dict[str, Optional[pd.DataFrame]] = {}
    overall_status = "completed";
    error_messages = []
    try:
        city_config_path = os.path.join(CITY_CONFIG_DIR, f"{city_name}.yaml")
        with open(city_config_path, 'r') as f:
            city_config = yaml.safe_load(f)
        scenario_config = get_scenario(scenario_id)
        target_ranges = scenario_config.get('target_ranges', {})
        if not target_ranges:
            target_ranges = {k.replace('default_target_', ''): v for k, v in city_config.items() if
                             k.startswith('default_target_')}
    except Exception as e:
        logger.error(f"Config loading failed for {city_name}/{scenario_id}: {e}", exc_info=True)
        return {"error": f"Config loading failed: {e}", "status": "error"}

    controllers_to_run_map = {
        'Baseline (Normal Setting)': ('baseline', None),
        'Baseline (Eco Setting)': ('baseline', None),
        'ML Normal': ('ml_normal', f"{city_name[0]}_normal"),
        'ML Eco': ('ml_eco', f"{city_name[0]}_eco")
    }
    for display_name, (controller_type_for_run, model_sfx_for_run) in controllers_to_run_map.items():
        hist_df, metrics_dict_run = run_single_simulation(
            city_name=city_name, scenario_id=scenario_id,
            controller_type=controller_type_for_run, controller_display_name=display_name,
            model_suffix=model_sfx_for_run)
        results[display_name] = metrics_dict_run if metrics_dict_run else {"error": "Sim failed: No metrics."}
        history_data[display_name] = hist_df
        if metrics_dict_run is None or metrics_dict_run.get("error") is not None:
            overall_status = "error";
            error_messages.append(f"{display_name}: {metrics_dict_run.get('error', 'Unknown error')}")

    plot_history_for_main_comparison = {
        'Baseline': history_data.get('Baseline (Normal Setting)'),
        'ML Normal': history_data.get('ML Normal'),
        'ML Eco': history_data.get('ML Eco')}
    plot_filename = create_comparison_figure(
        history_dfs=plot_history_for_main_comparison,
        scenario_id=scenario_id, city=city_name,
        target_ranges=target_ranges, save_dir=FIGURE_DIR)
    final_result = {
        "request": {"city": city_name, "scenario_id": scenario_id},
        "baseline_metrics": results.get('Baseline (Normal Setting)', {'error': 'Run skipped/failed'}),
        "ml_normal_metrics": results.get('ML Normal', {'error': 'Run skipped/failed'}),
        "ml_eco_metrics": results.get('ML Eco', {'error': 'Run skipped/failed'}),
        "plot_filename": plot_filename, "status": overall_status, "all_metrics_raw": results}
    if error_messages: final_result["error_message"] = "; ".join(error_messages)
    return final_result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run Greenhouse Full Comparison Evaluation Engine")
    parser.add_argument("--city", required=True, choices=['oslo', 'riyadh'], help="City name")
    parser.add_argument("--scenario", required=True, help="Scenario ID (e.g., oslo_summer_normal)")
    args = parser.parse_args()
    logger.info(f"Running evaluation directly for City: {args.city}, Scenario: {args.scenario}")
    comparison_data = run_full_comparison(args.city, args.scenario)
    logger.info("\n--- Full Comparison Results ---")
    import json

    print(json.dumps(comparison_data, indent=2, default=str))
    if comparison_data.get("plot_filename"):
        print(f"\nPlot saved to: {os.path.join(FIGURE_DIR, comparison_data['plot_filename'])}")
    else:
        print("\nPlot was not generated or failed.")

