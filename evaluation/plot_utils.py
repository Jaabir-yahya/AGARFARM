# evaluation/plot_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import logging
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__)

# Define a consistent color palette (example: colorblind friendly)
CONTROLLER_PALETTE = {
    "Baseline": sns.color_palette("colorblind")[0],
    "ML Normal": sns.color_palette("colorblind")[1],
    "ML Eco": sns.color_palette("colorblind")[2]
}

def setup_plotting_style(context='talk'):
    """Sets a consistent style for plots."""
    sns.set_style("ticks")
    sns.set_context(context) # 'paper', 'notebook', 'talk', 'poster'
    plt.rcParams['figure.figsize'] = (15, 7) # Default figure size
    plt.rcParams['figure.autolayout'] = True # Adjust layout automatically
    logger.info(f"Plotting style set to 'ticks', context '{context}'.")

def add_target_band(ax: plt.Axes, ticks: Any, min_val: Optional[float], max_val: Optional[float], label_prefix: str):
    """Helper to add target min/max lines and shaded band to a plot axis."""
    if min_val is not None:
        ax.axhline(min_val, color='gray', linestyle='--', lw=1.5, label=f'{label_prefix} Min ({min_val:.1f})')
    if max_val is not None:
        ax.axhline(max_val, color='gray', linestyle=':', lw=1.5, label=f'{label_prefix} Max ({max_val:.1f})')
    if min_val is not None and max_val is not None:
        # Use unique ticks for fill_between if possible, otherwise use axis limits
        try:
            unique_ticks = pd.unique(ticks)
            ax.fill_between(unique_ticks, min_val, max_val, color='gray', alpha=0.1, label='Target Band')
        except Exception: # Fallback if ticks aren't suitable
             ax.fill_between(ax.get_xlim(), min_val, max_val, color='gray', alpha=0.1)


def plot_sensor_comparison(
    combined_df: pd.DataFrame, # DataFrame with 'tick', sensor column, and 'Controller' column
    sensor: str,
    target_min: Optional[float],
    target_max: Optional[float],
    y_label: str,
    title: str,
    ax: plt.Axes,
    show_legend: bool = True
):
    """Plots a single sensor comparison with target bands on a given axis."""
    if combined_df.empty or sensor not in combined_df.columns:
        ax.text(0.5, 0.5, f'No data for {sensor}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title + " (No Data)")
        return

    sns.lineplot(data=combined_df, x='tick', y=sensor, hue='Controller',
                 palette=CONTROLLER_PALETTE, ax=ax, errorbar=None, legend=False) # Legend handled later

    add_target_band(ax, combined_df['tick'], target_min, target_max, sensor.upper())

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    # Legend will be added globally


def plot_resource_comparison(
    combined_df: pd.DataFrame,
    resource: str, # 'kwh' or 'water_l'
    y_label: str,
    title: str,
    ax: plt.Axes,
    show_legend: bool = True
):
    """Plots cumulative resource usage comparison on a given axis."""
    if combined_df.empty or resource not in combined_df.columns:
        ax.text(0.5, 0.5, f'No data for {resource}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title + " (No Data)")
        return

    sns.lineplot(data=combined_df, x='tick', y=resource, hue='Controller',
                 palette=CONTROLLER_PALETTE, ax=ax, errorbar=None, legend=False) # Legend handled later

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=0) # Resources shouldn't be negative


def create_comparison_figure(
    history_dfs: Dict[str, Optional[pd.DataFrame]], # Keys: 'Baseline', 'ML Normal', 'ML Eco'
    scenario_id: str,
    city: str,
    target_ranges: Dict[str, float],
    save_dir: str = "../figures" # Default save directory relative to this file
) -> Optional[str]:
    """
    Generates and saves a multi-panel comparison plot (PNG).

    Args:
        history_dfs: Dict where keys are controller display names and values are their history DataFrames.
        scenario_id: ID of the scenario being plotted.
        city: Name of the city ('oslo' or 'riyadh').
        target_ranges: Dictionary of target values for plotting bands.
        save_dir: Directory to save the plot image.

    Returns:
        Filename of the generated plot PNG (relative to save_dir), or None if plotting failed.
    """
    logger.info(f"Generating comparison plot for {city} - {scenario_id}...")
    setup_plotting_style() # Apply consistent style

    valid_dfs = {k: df for k, df in history_dfs.items() if df is not None and not df.empty}
    if not valid_dfs:
        logger.warning("No valid dataframes provided for plotting.")
        return None

    try:
        # Combine data for easier plotting
        all_dfs = []
        for controller_name, df in valid_dfs.items():
             df_copy = df.copy()
             df_copy['Controller'] = controller_name # Ensure consistent hue column name
             all_dfs.append(df_copy)
        if not all_dfs:
            logger.warning("No dataframes to combine for plotting.")
            return None
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Create figure with subplots
        num_plots = 5 # Temp, RH, SM, Energy, Water
        fig, axes = plt.subplots(num_plots, 1, figsize=(14, 6 * num_plots), sharex=True)
        fig.suptitle(f"Controller Comparison: {city.capitalize()} - {scenario_id}", fontsize=18, y=0.995)

        # Plot each variable
        plot_sensor_comparison(combined_df, 't_c', target_ranges.get('t_c_min'), target_ranges.get('t_c_max'), 'Temp (°C)', 'Temperature Control', axes[0], show_legend=False)
        plot_sensor_comparison(combined_df, 'rh', target_ranges.get('rh_min'), target_ranges.get('rh_max'), 'Humidity (%)', 'Relative Humidity Control', axes[1], show_legend=False)
        plot_sensor_comparison(combined_df, 'sm', target_ranges.get('sm_min'), target_ranges.get('sm_max'), 'Soil Moisture (%)', 'Soil Moisture Control', axes[2], show_legend=False)
        plot_resource_comparison(combined_df, 'kwh', 'Energy (kWh)', 'Cumulative Energy Usage', axes[3], show_legend=False)
        plot_resource_comparison(combined_df, 'water_l', 'Water (L)', 'Cumulative Water Usage', axes[4], show_legend=False)

        # Add shared X label
        axes[-1].set_xlabel("Simulation Tick")

        # Add a single legend for the whole figure
        # Get handles/labels from one of the plots that has them generated
        handles, labels = axes[0].get_legend_handles_labels()
        # Create unique legend entries (handles case where fill_between adds duplicate label)
        unique_handles_labels = {}
        for h, l in zip(handles, labels):
            if l not in unique_handles_labels:
                 unique_handles_labels[l] = h
        # Place legend outside the plots
        fig.legend(unique_handles_labels.values(), unique_handles_labels.keys(),
                   loc='upper right', bbox_to_anchor=(0.99, 0.98), title="Controller/Targets")

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust rect to prevent suptitle overlap

        # Save plot
        os.makedirs(save_dir, exist_ok=True) # Ensure save directory exists
        plot_filename = f"comparison_{city}_{scenario_id}.png"
        plot_filepath = os.path.join(save_dir, plot_filename)
        plt.savefig(plot_filepath, bbox_inches='tight')
        plt.close(fig) # Close the figure to free memory
        logger.info(f"Comparison plot saved to {plot_filepath}")
        return plot_filename # Return just the filename

    except Exception as e:
        logger.error(f"ERROR generating comparison plot for {scenario_id}: {e}", exc_info=True)
        if 'fig' in locals() and fig: plt.close(fig) # Attempt to close figure on error
        return None
