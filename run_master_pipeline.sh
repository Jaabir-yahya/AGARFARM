#!/bin/bash

# This script automates the full pipeline:
# 1. Trains all 4 PPO models (Oslo/Normal, Oslo/Eco, Riyadh/Normal, Riyadh/Eco).
# 2. Runs evaluations for predefined scenarios using these models and baselines.
# 3. Generates a consolidated diagnostic report.



echo "--- AGARTECHdiss Master Pipeline Started ---"
START_TIME_PIPELINE=$(date +%s)
TIMESTAMP_START=$(date +"%Y-%m-%d %H:%M:%S")
echo "Start Timestamp: $TIMESTAMP_START"
echo ""


echo "--- Phase 1: Training ML Models ---"

# Training Oslo Normal
echo "[TRAINING] Starting: Oslo Normal..."
python ml_training/train_ppo.py --city oslo --mode normal
echo "[TRAINING] Finished: Oslo Normal."
echo ""

# Training Oslo Eco
echo "[TRAINING] Starting: Oslo Eco..."
python ml_training/train_ppo.py --city oslo --mode eco
echo "[TRAINING] Finished: Oslo Eco."
echo ""

# Training Riyadh Normal
echo "[TRAINING] Starting: Riyadh Normal..."
python ml_training/train_ppo.py --city riyadh --mode normal
echo "[TRAINING] Finished: Riyadh Normal."
echo ""

# Training Riyadh Eco
echo "[TRAINING] Starting: Riyadh Eco..."
python ml_training/train_ppo.py --city riyadh --mode eco
echo "[TRAINING] Finished: Riyadh Eco."
echo ""

echo "--- Phase 1: ML Model Training Attempted for all cases ---"
echo ""

# --- Phase 2: Generating Final Evaluation Report ---
echo "--- Phase 2: Generating Final Evaluation Report ---"
# This Python script will run evaluations and compile the diagnostics
python generate_final_report.py
echo "--- Phase 2: Final Evaluation Report Generation Attempted ---"
echo ""

END_TIME_PIPELINE=$(date +%s)
TIMESTAMP_END=$(date +"%Y-%m-%d %H:%M:%S")
DURATION_PIPELINE=$((END_TIME_PIPELINE - START_TIME_PIPELINE))

echo "--- AGARTECHdiss Master Pipeline Finished ---"
echo "End Timestamp: $TIMESTAMP_END"
echo "Total Pipeline Duration: $DURATION_PIPELINE seconds ($((DURATION_PIPELINE / 60)) minutes)."
echo "Please check 'final_project_diagnostics.txt' for the summary."
echo "Check 'figures/' directory for generated plots."
echo "Check 'logs/ppo_tensorboard/' for TensorBoard logs."

