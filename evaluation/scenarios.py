# Defines evaluation scenarios and necessary constants

import os
import json # For the if __name__ == '__main__' block

# --- Constants for Evaluation ---
EVAL_DT_MIN = 5

# --- OBSERVATION FEATURE ORDER & RELATED CONSTANTS (28 Features) ---
N_ACTUATOR_HISTORY_STEPS = 3

ACTUATORS = ['fan', 'ac', 'vent', 'irrigation'] # Now uppercase
N_ACTUATORS = len(ACTUATORS) # Explicitly define N_ACTUATORS

# Base features (16) - Order must match training environment
BASE_OBSERVATION_FEATURE_ORDER = [
    't_c', 'rh', 'sm', 'vpd', 'rain_tank_l', 'outside_temp', 'outside_rh',
    'sin_hour', 'cos_hour', 'is_raining',
    'target_t_c_min', 'target_t_c_max', 'target_rh_min',
    'target_rh_max', 'target_sm_min', 'target_sm_max',
]

# History features (4 actuators * 3 steps = 12 features)
HISTORY_FEATURES_ORDERED = []
for i in range(1, N_ACTUATOR_HISTORY_STEPS + 1):
     for act_name in ACTUATORS: # Use the renamed uppercase variable
          HISTORY_FEATURES_ORDERED.append(f"{act_name}_on_t_minus_{i}")

OBSERVATION_FEATURE_ORDER = BASE_OBSERVATION_FEATURE_ORDER + HISTORY_FEATURES_ORDERED

# --- Check Length ---
EXPECTED_OBS_DIM = 16 + (N_ACTUATORS * N_ACTUATOR_HISTORY_STEPS) # Use uppercase N_ACTUATORS
if len(OBSERVATION_FEATURE_ORDER) != EXPECTED_OBS_DIM:
    raise ValueError(
        f"OBSERVATION_FEATURE_ORDER length is {len(OBSERVATION_FEATURE_ORDER)}, "
        f"expected {EXPECTED_OBS_DIM} (16 base + {N_ACTUATORS * N_ACTUATOR_HISTORY_STEPS} history)."
    )

# --- Default target ranges ---
TARGETS_OSLO_DEFAULT = {
    't_c_min': 16.0, 't_c_max': 20.0, 'rh_min': 65.0, 'rh_max': 75.0, 'sm_min': 50.0, 'sm_max': 65.0
}
TARGETS_RIYADH_DEFAULT = {
    't_c_min': 22.0, 't_c_max': 28.0, 'rh_min': 40.0, 'rh_max': 55.0, 'sm_min': 35.0, 'sm_max': 50.0
}

# --- Simulation duration for evaluations ---
EVAL_DURATION_DAYS = 5
EVAL_DURATION_STEPS = int(EVAL_DURATION_DAYS * 24 * 60 / EVAL_DT_MIN)

# --- Scenario Definitions ---
SCENARIOS = {
    "oslo_summer_normal": {
        "description": "Oslo: Typical summer day (July), standard targets.", "city": "oslo", "sim_month": 7,
        "duration_steps": EVAL_DURATION_STEPS, "target_ranges": TARGETS_OSLO_DEFAULT, "initial_state": None,
        "is_maintenance": False, "offline_only": False,
    },
    "oslo_winter_heating": {
        "description": "Oslo: Winter day (Jan), heating challenge.", "city": "oslo", "sim_month": 1,
        "duration_steps": EVAL_DURATION_STEPS, "target_ranges": TARGETS_OSLO_DEFAULT, "initial_state": {"t_c": 15.0},
        "is_maintenance": False, "offline_only": False,
    },
     "oslo_maintenance_recovery": {
        "description": "Oslo: Simulates recovery after a maintenance override period.", "city": "oslo", "sim_month": 9,
        "duration_steps": EVAL_DURATION_STEPS, "target_ranges": TARGETS_OSLO_DEFAULT, "initial_state": None,
        "is_maintenance": True, "override_start_step": EVAL_DURATION_STEPS // 4,
        "override_duration_steps": EVAL_DURATION_STEPS // 8,
        "override_actuator_states": {"fan": False, "ac": False, "vent": True, "irrigation": False},
        "offline_only": True,
    },
    "riyadh_summer_cooling": {
        "description": "Riyadh: Typical summer day (July), cooling challenge.", "city": "riyadh", "sim_month": 7,
        "duration_steps": EVAL_DURATION_STEPS, "target_ranges": TARGETS_RIYADH_DEFAULT, "initial_state": None,
        "is_maintenance": False, "offline_only": False,
    },
    "riyadh_winter_normal": {
         "description": "Riyadh: Typical winter day (Jan), standard targets.", "city": "riyadh", "sim_month": 1,
         "duration_steps": EVAL_DURATION_STEPS, "target_ranges": TARGETS_RIYADH_DEFAULT, "initial_state": None,
         "is_maintenance": False, "offline_only": False,
    },
    "riyadh_winter_low_water": {
         "description": "Riyadh: Winter day starting with low water tank.", "city": "riyadh", "sim_month": 1,
         "duration_steps": EVAL_DURATION_STEPS, "target_ranges": TARGETS_RIYADH_DEFAULT,
         "initial_state": {"rain_tank_l": 20.0}, "is_maintenance": False, "offline_only": False,
    },
}

def get_scenario(scenario_id: str):
    if scenario_id not in SCENARIOS:
        for key in SCENARIOS:
            if key.lower() == scenario_id.lower():
                return SCENARIOS[key]
        raise ValueError(f"Unknown scenario ID: {scenario_id}")
    return SCENARIOS[scenario_id]

def list_scenarios(include_offline: bool = False):
    return [
        {"id": k, "description": v["description"]}
        for k, v in SCENARIOS.items()
        if include_offline or not v.get("offline_only", False)
    ]

if __name__ == '__main__':
    print("--- evaluation/scenarios.py self-check ---")
    print(f"EVAL_DT_MIN: {EVAL_DT_MIN}")
    print(f"N_ACTUATOR_HISTORY_STEPS: {N_ACTUATOR_HISTORY_STEPS}")
    print(f"ACTUATORS: {ACTUATORS}") # Should show uppercase
    print(f"N_ACTUATORS: {N_ACTUATORS}") # Should show 4
    print(f"OBSERVATION_FEATURE_ORDER Length: {len(OBSERVATION_FEATURE_ORDER)}") # Should show 28
    print(f"EXPECTED_OBS_DIM calculated: {EXPECTED_OBS_DIM}") # Should show 28
    print("--- Self-check complete ---")