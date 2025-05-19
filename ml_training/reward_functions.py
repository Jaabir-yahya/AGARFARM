import math
from typing import Dict, Any

# --- Attempt to import simulator types for type hinting ---
try:
    import sys, os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from simulator.core import SensorSnapshot, ControlState
except ImportError:
    SensorSnapshot = Any
    ControlState    = Any
    print("WARN: Using fallback types in reward_functions.py")

def calculate_reward(
    state: SensorSnapshot,
    action: ControlState,
    next_state: SensorSnapshot,
    targets: Dict[str, float],
    resources_used_step: Dict[str, float],
    reward_weights: Dict[str, float]
) -> float:
    """
    Reward = −(deviation penalties + resource penalties) + in-band bonus
    """
    # --- Weights & Buffers ---
    w_t_dev = reward_weights.get('W_T_DEV', 1.0)
    w_rh_dev= reward_weights.get('W_RH_DEV', 0.5)
    w_sm_dev= reward_weights.get('W_SM_DEV', 0.8)
    w_kwh   = reward_weights.get('W_KWH', 0.1)
    w_water = reward_weights.get('W_WATER', 0.05)
    w_bonus = reward_weights.get('W_IN_BAND_BONUS', 0.05)

    temp_buf = reward_weights.get('TEMP_BUFFER', 0.5)
    rh_buf   = reward_weights.get('RH_BUFFER', 3.0)
    sm_buf   = reward_weights.get('SM_BUFFER', 3.0)

    # --- Temperature Deviation ---
    t_pen = 0.0
    low_t = targets['t_c_min'] - temp_buf
    high_t= targets['t_c_max'] + temp_buf
    if next_state.t_c > high_t:
        t_pen = w_t_dev * (next_state.t_c - high_t)**2
    elif next_state.t_c < low_t:
        t_pen = w_t_dev * (low_t - next_state.t_c)**2

    # --- Humidity Deviation ---
    rh_pen = 0.0
    low_rh = targets['rh_min'] - rh_buf
    high_rh= targets['rh_max'] + rh_buf
    if next_state.rh > high_rh:
        rh_pen = w_rh_dev * (next_state.rh - high_rh)**2
    elif next_state.rh < low_rh:
        rh_pen = w_rh_dev * (low_rh - next_state.rh)**2

    # --- Soil Moisture Deviation ---
    sm_pen = 0.0
    low_sm = targets['sm_min'] - sm_buf
    high_sm= targets['sm_max'] + sm_buf
    if next_state.sm > high_sm:
        sm_pen = w_sm_dev * (next_state.sm - high_sm)**2
    elif next_state.sm < low_sm:
        sm_pen = w_sm_dev * (low_sm - next_state.sm)**2

    # --- Resource Penalties ---
    kwh_pen   = w_kwh   * resources_used_step.get('kwh', 0.0)
    water_pen = w_water * resources_used_step.get('water_l', 0.0)

    # --- In-Band Bonus ---
    bonus = 0.0
    in_t = targets['t_c_min'] <= next_state.t_c <= targets['t_c_max']
    in_r = targets['rh_min']  <= next_state.rh  <= targets['rh_max']
    in_s = targets['sm_min']  <= next_state.sm  <= targets['sm_max']
    if in_t and in_r and in_s:
        bonus = w_bonus

    total_penalty = t_pen + rh_pen + sm_pen + kwh_pen + water_pen
    return float(-total_penalty + bonus)
