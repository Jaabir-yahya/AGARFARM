import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def calculate_metrics(history_df: pd.DataFrame, target_ranges: Dict[str, float]) -> Dict[str, Optional[float]]:
    """
    Calculates key performance indicators from simulation history DataFrame.

    Args:
        history_df: DataFrame containing simulation logs (tick, t_c, rh, sm, kwh, water_l, etc.).
                    Should have columns matching sensor names and resource names.
        target_ranges: Dictionary with keys like 't_c_min', 't_c_max', 'rh_min', 'rh_max', etc.

    Returns:
        Dictionary containing calculated metrics. Returns None for values if calculation fails
        or data is missing. Includes an 'error' key if major calculation fails.
    """
    metrics: Dict[str, Optional[float]] = {}
    required_cols = ['t_c', 'rh', 'sm', 'vpd', 'kwh', 'water_l', 'fan', 'ac', 'vent', 'irrigation']

    if history_df is None or history_df.empty:
        metrics["error"] = 1.0 # Use a numerical value to indicate error state
        print("WARN: Cannot calculate metrics, history_df is empty or None.")
        # Return default None values for expected keys
        for key in ['mean_t_c', 'mean_rh', 'mean_sm', 'mean_vpd', 'pct_time_t_c_out',
                    'pct_time_rh_out', 'pct_time_sm_out', 'mae_t_c', 'mae_rh', 'mae_sm',
                    'total_kwh', 'total_water_l', 'avg_usage_fan', 'avg_usage_ac',
                    'avg_usage_vent', 'avg_usage_irrigation']:
            metrics[key] = None
        return metrics

    # Check for essential columns
    missing_cols = [col for col in required_cols if col not in history_df.columns]
    if missing_cols:
        metrics["error"] = 1.0
        print(f"WARN: Missing required columns for metrics calculation: {missing_cols}")
        # Return default None values
        for key in ['mean_t_c', 'mean_rh', 'mean_sm', 'mean_vpd', 'pct_time_t_c_out',
                    'pct_time_rh_out', 'pct_time_sm_out', 'mae_t_c', 'mae_rh', 'mae_sm',
                    'total_kwh', 'total_water_l', 'avg_usage_fan', 'avg_usage_ac',
                    'avg_usage_vent', 'avg_usage_irrigation']:
            metrics[key] = None
        return metrics


    try:
        # Basic Averages
        metrics['mean_t_c'] = history_df['t_c'].mean()
        metrics['mean_rh'] = history_df['rh'].mean()
        metrics['mean_sm'] = history_df['sm'].mean()
        metrics['mean_vpd'] = history_df['vpd'].mean()

        # Time Outside Bands (%)
        n_ticks = len(history_df)
        if n_ticks > 0:
            t_out = history_df[(history_df['t_c'] < target_ranges['t_c_min']) | (history_df['t_c'] > target_ranges['t_c_max'])]
            rh_out = history_df[(history_df['rh'] < target_ranges['rh_min']) | (history_df['rh'] > target_ranges['rh_max'])]
            sm_out = history_df[(history_df['sm'] < target_ranges['sm_min']) | (history_df['sm'] > target_ranges['sm_max'])]
            metrics['pct_time_t_c_out'] = (len(t_out) / n_ticks) * 100
            metrics['pct_time_rh_out'] = (len(rh_out) / n_ticks) * 100
            metrics['pct_time_sm_out'] = (len(sm_out) / n_ticks) * 100
        else:
             metrics['pct_time_t_c_out'] = 0.0
             metrics['pct_time_rh_out'] = 0.0
             metrics['pct_time_sm_out'] = 0.0


        # Mean Absolute Error (MAE) from Target Band
        # Calculate deviation only when outside the band
        t_dev = ((history_df['t_c'] - target_ranges['t_c_max']).clip(lower=0) + \
                 (target_ranges['t_c_min'] - history_df['t_c']).clip(lower=0))
        rh_dev = ((history_df['rh'] - target_ranges['rh_max']).clip(lower=0) + \
                  (target_ranges['rh_min'] - history_df['rh']).clip(lower=0))
        sm_dev = ((history_df['sm'] - target_ranges['sm_max']).clip(lower=0) + \
                  (target_ranges['sm_min'] - history_df['sm']).clip(lower=0))

        metrics['mae_t_c'] = t_dev.mean()
        metrics['mae_rh'] = rh_dev.mean()
        metrics['mae_sm'] = sm_dev.mean()


        # Resource Consumption (Total - use max value which represents the end)
        metrics['total_kwh'] = history_df['kwh'].max()
        metrics['total_water_l'] = history_df['water_l'].max()


        # Actuator Usage (Average Fraction ON)
        actuators = ['fan', 'ac', 'vent', 'irrigation']
        for act in actuators:
            metrics[f'avg_usage_{act}'] = history_df[act].mean()

        # Replace NaN with None for JSON compatibility if needed
        metrics_clean = {k: (None if pd.isna(v) else v) for k, v in metrics.items()}
        return metrics_clean

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        # Return default None values for expected keys
        metrics_err: Dict[str, Optional[float]] = {"error": 1.0}
        for key in ['mean_t_c', 'mean_rh', 'mean_sm', 'mean_vpd', 'pct_time_t_c_out',
                    'pct_time_rh_out', 'pct_time_sm_out', 'mae_t_c', 'mae_rh', 'mae_sm',
                    'total_kwh', 'total_water_l', 'avg_usage_fan', 'avg_usage_ac',
                    'avg_usage_vent', 'avg_usage_irrigation']:
            metrics_err[key] = None
        return metrics_err