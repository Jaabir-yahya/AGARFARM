#!/bin/bash

# Create main directories
mkdir -p notebooks simulator/city_configs controllers ml_training/config evaluation/trained_models figures

# Create empty Python module initializers
touch simulator/__init__.py
touch controllers/__init__.py
touch ml_training/__init__.py
touch evaluation/__init__.py

# Create empty Python files
touch simulator/core.py
touch controllers/baseline.py
touch controllers/smart_ml_agent.py
touch ml_training/custom_env_wrapper.py
touch ml_training/reward_functions.py
touch evaluation/scenarios.py
touch evaluation/metrics.py
touch evaluation/plot_utils.py  # Optional helper

# Create empty YAML config files
touch simulator/city_configs/oslo.yaml
touch simulator/city_configs/riyadh.yaml
touch ml_training/config/oslo_normal_hparams.yaml
touch ml_training/config/oslo_eco_hparams.yaml
touch ml_training/config/riyadh_normal_hparams.yaml
touch ml_training/config/riyadh_eco_hparams.yaml

# Create empty Notebook files
touch notebooks/01_Simulator_Baseline_Dev.ipynb
touch notebooks/02_ML_Environment_Wrapper.ipynb
touch notebooks/03_Train_ML_Models.ipynb
touch notebooks/04_Evaluate_Controllers.ipynb

# Create requirements and README
touch requirements.txt
touch README.md

echo "✅ File structure created successfully."