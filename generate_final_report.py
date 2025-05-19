import os
import sys
import yaml
import pandas as pd
from datetime import datetime
import traceback
import logging

# --- Setup Project Root Path ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_script_dir  # Assuming this script is in the project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Imports from project ---
try:
    from evaluation.run_evaluation_engine import run_full_comparison, MODEL_DIR as EVAL_MODEL_DIR, FIGURE_DIR
    from evaluation.scenarios import list_scenarios, get_scenario
    from evaluation.metrics import calculate_metrics  # For potential direct use if needed
except ImportError as e:
    print(f"ERROR: Cannot import project modules in generate_final_report.py: {e}")
    print(f"Current sys.path: {sys.path}")
    print("Ensure this script is run from the project root directory.")
    sys.exit(1)

# --- Configuration ---
OUTPUT_REPORT_FILENAME = "final_project_diagnostics.txt"
CASES_TO_TRAIN = [('oslo', 'normal'), ('oslo', 'eco'), ('riyadh', 'normal'), ('riyadh', 'eco')]
# Define scenarios to include in the report (e.g., all non-offline scenarios)
SCENARIOS_TO_EVALUATE = [s['id'] for s in list_scenarios(include_offline=False)]

# Configure logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_training_files():
    """Checks for the existence of trained model and normalization files."""
    report_lines = ["\n--- ML Model Training File Check ---"]
    all_found = True
    for city, mode in CASES_TO_TRAIN:
        city_code = city[0]
        model_suffix = f"{city_code}_{mode}"
        model_filename = f"ppo_agent_{model_suffix}.zip"
        vec_norm_filename = f"vecnormalize_{model_suffix}.pkl"
        model_filepath = os.path.join(EVAL_MODEL_DIR, model_filename)
        vec_norm_filepath = os.path.join(EVAL_MODEL_DIR, vec_norm_filename)

        model_found = os.path.exists(model_filepath)
        stats_found = os.path.exists(vec_norm_filepath)
        report_lines.append(f"\nCase: {city.capitalize()} {mode.capitalize()}")
        report_lines.append(f"  - Model (.zip): {'Found' if model_found else 'MISSING'} ({model_filepath})")
        report_lines.append(f"  - Stats (.pkl): {'Found' if stats_found else 'MISSING'} ({vec_norm_filepath})")
        if not model_found or not stats_found:
            all_found = False

    if all_found:
        report_lines.append("\nStatus: All expected model and stats files appear to be present.")
    else:
        report_lines.append(
            "\nWARNING: Some model or stats files are MISSING. Training may not have completed successfully.")
    return report_lines, all_found


def run_all_evaluations():
    """Runs evaluations for all specified scenarios and collects results."""
    report_lines = ["\n\n--- Evaluation Results Summary ---"]
    all_evaluation_metrics = {}  # Store metrics for all scenarios and controllers
    generated_plots = []

    if not SCENARIOS_TO_EVALUATE:
        report_lines.append("No scenarios defined for evaluation.")
        return report_lines, all_evaluation_metrics, generated_plots

    for scenario_id in SCENARIOS_TO_EVALUATE:
        try:
            scenario_config = get_scenario(scenario_id)
            city = scenario_config['city']
            report_lines.append(f"\n\n--- Evaluating Scenario: {scenario_id} (City: {city.capitalize()}) ---")
            logger.info(f"Running full comparison for: {city} / {scenario_id}")

            comparison_results = run_full_comparison(city_name=city, scenario_id=scenario_id)
            all_evaluation_metrics[scenario_id] = comparison_results

            # Add metrics to report
            if comparison_results.get("status") == "completed":
                report_lines.append("  Status: COMPLETED")
            else:
                report_lines.append(f"  Status: {comparison_results.get('status', 'UNKNOWN_ERROR')}")
                if comparison_results.get("error_message"):
                    report_lines.append(f"  Error Message: {comparison_results.get('error_message')}")

            # Baseline (Normal Setting) Metrics
            bl_norm_metrics = comparison_results.get('baseline_metrics', {})
            report_lines.append("\n  Baseline (Normal Setting) Metrics:")
            if bl_norm_metrics and bl_norm_metrics.get('error') is None:
                for k, v in bl_norm_metrics.items():
                    if k != 'error': report_lines.append(
                        f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")
            else:
                report_lines.append(f"    Error: {bl_norm_metrics.get('error', 'N/A')}")

            # ML Normal Metrics
            ml_norm_metrics = comparison_results.get('ml_normal_metrics', {})
            report_lines.append("\n  ML Normal Metrics:")
            if ml_norm_metrics and ml_norm_metrics.get('error') is None:
                for k, v in ml_norm_metrics.items():
                    if k != 'error': report_lines.append(
                        f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")
            else:
                report_lines.append(f"    Error: {ml_norm_metrics.get('error', 'N/A')}")

            # ML Eco Metrics
            ml_eco_metrics = comparison_results.get('ml_eco_metrics', {})
            report_lines.append("\n  ML Eco Metrics:")
            if ml_eco_metrics and ml_eco_metrics.get('error') is None:
                for k, v in ml_eco_metrics.items():
                    if k != 'error': report_lines.append(
                        f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")
            else:
                report_lines.append(f"    Error: {ml_eco_metrics.get('error', 'N/A')}")

            # Resource Savings
            report_lines.append("\n  Resource Savings (vs Baseline Normal):")
            if bl_norm_metrics and bl_norm_metrics.get('total_kwh') is not None and bl_norm_metrics.get(
                    'error') is None:
                base_kwh = bl_norm_metrics['total_kwh']
                base_h2o = bl_norm_metrics['total_water_l']

                if ml_norm_metrics and ml_norm_metrics.get('total_kwh') is not None and ml_norm_metrics.get(
                        'error') is None:
                    norm_kwh_sav = base_kwh - ml_norm_metrics['total_kwh']
                    norm_h2o_sav = base_h2o - ml_norm_metrics['total_water_l']
                    norm_kwh_pct = (norm_kwh_sav / base_kwh * 100) if base_kwh != 0 else (
                        0 if norm_kwh_sav == 0 else float('inf'))
                    norm_h2o_pct = (norm_h2o_sav / base_h2o * 100) if base_h2o != 0 else (
                        0 if norm_h2o_sav == 0 else float('inf'))
                    report_lines.append(
                        f"    - ML Normal: Energy={norm_kwh_sav:.2f} kWh ({norm_kwh_pct:.1f}%), Water={norm_h2o_sav:.2f} L ({norm_h2o_pct:.1f}%)")
                else:
                    report_lines.append("    - ML Normal: Metrics unavailable for savings calculation.")

                if ml_eco_metrics and ml_eco_metrics.get('total_kwh') is not None and ml_eco_metrics.get(
                        'error') is None:
                    eco_kwh_sav = base_kwh - ml_eco_metrics['total_kwh']
                    eco_h2o_sav = base_h2o - ml_eco_metrics['total_water_l']
                    eco_kwh_pct = (eco_kwh_sav / base_kwh * 100) if base_kwh != 0 else (
                        0 if eco_kwh_sav == 0 else float('inf'))
                    eco_h2o_pct = (eco_h2o_sav / base_h2o * 100) if base_h2o != 0 else (
                        0 if eco_h2o_sav == 0 else float('inf'))
                    report_lines.append(
                        f"    - ML Eco:    Energy={eco_kwh_sav:.2f} kWh ({eco_kwh_pct:.1f}%), Water={eco_h2o_sav:.2f} L ({eco_h2o_pct:.1f}%)")
                else:
                    report_lines.append("    - ML Eco:    Metrics unavailable for savings calculation.")
            else:
                report_lines.append("    - Baseline (Normal Setting) metrics unavailable for savings calculation.")

            plot_file = comparison_results.get("plot_filename")
            if plot_file:
                full_plot_path = os.path.join(FIGURE_DIR, plot_file)
                report_lines.append(f"  Plot Generated: {full_plot_path}")
                generated_plots.append(full_plot_path)
            else:
                report_lines.append("  Plot Generation: Failed or not available for this scenario.")

        except Exception as e:
            error_msg = f"ERROR during evaluation of scenario '{scenario_id}': {e}"
            logger.error(error_msg, exc_info=True)
            report_lines.append(f"\n  {error_msg}")
            traceback_str = traceback.format_exc()
            report_lines.append(f"  Traceback: {traceback_str}")
            all_evaluation_metrics[scenario_id] = {"error": error_msg}  # Store error

    return report_lines, all_evaluation_metrics, generated_plots


def main():
    """Main function to generate the full diagnostic report."""
    logger.info("Starting Final Report Generation...")
    report_content = [f"AGARTECHdiss - Final Project Diagnostics Report"]
    report_content.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Check Training Files
    training_file_lines, all_models_found = check_training_files()
    report_content.extend(training_file_lines)

    if not all_models_found:
        report_content.append(
            "\n\nWARNING: Not all ML models were found. Evaluation results may be incomplete or rely on older models.")

    # 2. Run Evaluations and Collect Data
    evaluation_lines, eval_metrics_data, plot_list = run_all_evaluations()
    report_content.extend(evaluation_lines)

    report_content.append("\n\n--- Generated Plot Files ---")
    if plot_list:
        for plot_path in plot_list:
            report_content.append(f"- {plot_path}")
    else:
        report_content.append("No plots were generated during this run.")

    report_content.append("\n\n--- End of Report ---")

    # Write report to file
    try:
        with open(OUTPUT_REPORT_FILENAME, "w") as f:
            f.write("\n".join(report_content))
        logger.info(f"Diagnostic report saved to: {OUTPUT_REPORT_FILENAME}")
    except Exception as e:
        logger.error(f"Failed to write diagnostic report: {e}", exc_info=True)


if __name__ == "__main__":
    main()

