# Notebook 3: Train ML Models (PPO)

**Goal:** Train the four specialized PPO models (Oslo/Riyadh x Normal/Eco) using Stable Baselines3 and save the trained agents and normalization statistics.

**Note:** Full training runs can take a very long time (hours). It's recommended to run short tests (e.g., 10k-50k steps) in the notebook first to ensure everything works, then run the full training (e.g., 2M steps) potentially as a separate Python script (`ml_training/train_ppo.py`) or overnight in the notebook.

## 1. Setup and Imports

Import necessary libraries from Stable Baselines3, Gymnasium, YAML, etc., and the custom `GreenhouseEnv` wrapper.


```
# Cell 1: Setup and Imports
import sys
import os
import yaml
import time
import gymnasium as gym
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import traceback
import pandas as pd # Needed for diagnostics cell
import numpy as np # Needed for diagnostics cell

# --- Add project root to sys.path ---
# This allows importing modules from sibling directories (simulator, controllers, ml_training)
notebook_dir = os.getcwd() # Should be AGARTECHdiss_simplified/notebooks
project_root = os.path.abspath(os.path.join(notebook_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added '{project_root}' to sys.path")

# --- Import project modules ---
try:
    # Import the environment wrapper itself
    from ml_training.custom_env_wrapper import GreenhouseEnv # The wrapper developed in Notebook 02
    print("Successfully imported project modules.")
except ImportError as e:
    print(f"ERROR: Failed to import modules: {e}")
    print("Ensure you have run Notebook 02 successfully and created necessary files.")
    print("Make sure stable_baselines3 and gymnasium are installed (`pip install -r requirements.txt`).")
    # Define dummy class if import fails, to potentially allow later cells to run without NameError
    if 'GreenhouseEnv' not in locals():
        class GreenhouseEnv: pass # Dummy

# --- Check SB3 Version ---
print(f"Using Stable-Baselines3 Version: {stable_baselines3.__version__}")
print(f"Using Gymnasium Version: {gym.__version__}")
```

    Successfully imported project modules.
    Using Stable-Baselines3 Version: 2.6.0
    Using Gymnasium Version: 1.1.1


## 2. Configuration and Paths

Define paths for loading configurations and saving models, logs, and checkpoints. Set training parameters.


```
# Cell 2: Configuration and Paths

# --- Define Base Paths ---
# Using absolute paths based on project_root is safer when running notebooks
MODEL_SAVE_DIR = os.path.abspath(os.path.join(project_root, "evaluation", "trained_models"))
CITY_CONFIG_DIR = os.path.abspath(os.path.join(project_root, "simulator", "city_configs"))
HPARAM_CONFIG_DIR = os.path.abspath(os.path.join(project_root, "ml_training", "config"))
# Save logs outside notebooks dir for clarity
TENSORBOARD_LOG_DIR_BASE = os.path.abspath(os.path.join(project_root, "logs", "ppo_tensorboard"))
CHECKPOINT_DIR_BASE = os.path.abspath(os.path.join(project_root, "logs", "ppo_checkpoints"))

# Ensure directories exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR_BASE, exist_ok=True)
os.makedirs(CHECKPOINT_DIR_BASE, exist_ok=True)

print(f"Model Save Directory: {MODEL_SAVE_DIR}")
print(f"Tensorboard Base Directory: {TENSORBOARD_LOG_DIR_BASE}")
print(f"Checkpoint Base Directory: {CHECKPOINT_DIR_BASE}")
print(f"City Config Directory: {CITY_CONFIG_DIR}")
print(f"Hyperparameter Directory: {HPARAM_CONFIG_DIR}")

# --- Training Parameters ---
# Set to True for full runs, False for quick debug runs
FULL_TRAINING_RUN = False # <<< SET TO True FOR FINAL TRAINING RUNS

if FULL_TRAINING_RUN:
    # Use values suitable for full runs (can still be overridden by hparams)
    DEFAULT_TOTAL_TIMESTEPS = 2_000_000
    DEFAULT_NUM_CPU = 4 # Adjust based on your machine's cores
    print("\n--- CONFIGURATION: FULL TRAINING RUN ---")
else:
    # Short run for debugging
    DEFAULT_TOTAL_TIMESTEPS = 10000 # Adjust as needed for a quick test (~1 episode)
    DEFAULT_NUM_CPU = 1 # Use single process for easier debugging
    print("\n--- CONFIGURATION: SHORT DEBUG RUN ---")
```

    Model Save Directory: /Users/jaabirahmed/Documents/AGARFARM/evaluation/trained_models
    Tensorboard Base Directory: /Users/jaabirahmed/Documents/AGARFARM/logs/ppo_tensorboard
    Checkpoint Base Directory: /Users/jaabirahmed/Documents/AGARFARM/logs/ppo_checkpoints
    City Config Directory: /Users/jaabirahmed/Documents/AGARFARM/simulator/city_configs
    Hyperparameter Directory: /Users/jaabirahmed/Documents/AGARFARM/ml_training/config
    
    --- CONFIGURATION: SHORT DEBUG RUN ---


## 3. Training Function Definition

Define a function to encapsulate the training process for a specific city and mode. This makes it reusable for all four training runs.


```
# Cell 3: Training Function Definition

def train_model(city: str, mode: str, total_timesteps: int, num_cpu: int):
    """Trains and saves a PPO model for a given city and mode."""
    print(f"\n--- Starting Training: {city.capitalize()} {mode.capitalize()} ---")
    start_time = time.time()
    training_successful = False # Flag to track success
    env = None # Initialize env to None
    model = None # Initialize model to None

    # --- Construct Paths ---
    hparam_config_file = os.path.join(HPARAM_CONFIG_DIR, f"{city}_{mode}_hparams.yaml")
    city_config_file = os.path.join(CITY_CONFIG_DIR, f"{city}.yaml")
    model_save_prefix = f"ppo_agent_{city[0]}_{mode}" # e.g., ppo_agent_o_normal
    model_save_path = os.path.join(MODEL_SAVE_DIR, model_save_prefix) # SB3 adds .zip
    vec_norm_save_path = os.path.join(MODEL_SAVE_DIR, f"vecnormalize_{city[0]}_{mode}.pkl") # Save stats in same dir
    tensorboard_log_path = os.path.join(TENSORBOARD_LOG_DIR_BASE, f"{city}_{mode}")
    checkpoint_log_path = os.path.join(CHECKPOINT_DIR_BASE, f"{city}_{mode}")
    os.makedirs(checkpoint_log_path, exist_ok=True) # Ensure checkpoint dir exists

    print(f" HParams File: {hparam_config_file}")
    print(f" City Config: {city_config_file}")
    print(f" Model Save Prefix: {model_save_path}")
    print(f" VecNormalize Save Path: {vec_norm_save_path}")
    print(f" Tensorboard Logs: {tensorboard_log_path}")
    print(f" Checkpoints: {checkpoint_log_path}")

    # --- Load Configs ---
    if not os.path.exists(hparam_config_file): raise FileNotFoundError(f"HParam file missing: {hparam_config_file}")
    if not os.path.exists(city_config_file): raise FileNotFoundError(f"City config missing: {city_config_file}")

    with open(hparam_config_file, 'r') as f: hparams = yaml.safe_load(f)
    with open(city_config_file, 'r') as f: city_cfg_data = yaml.safe_load(f)
    target_ranges = {k.replace('default_target_',''): v for k, v in city_cfg_data.items() if k.startswith('default_target_')}

    # Use total_timesteps and num_cpu passed into this function
    actual_num_cpu = max(1, min(num_cpu, os.cpu_count()))

    print(f" Training for {total_timesteps} steps using {actual_num_cpu} CPUs.")

    # --- Create Environment ---
    try:
        if GreenhouseEnv.__name__ == 'type': # Check if the dummy class is still defined
            raise ImportError("GreenhouseEnv not properly imported.")

        env_kwargs = {
            'city_config_path': city_config_file,
            'target_ranges': target_ranges,
            'mode': mode,
            'dt_min': hparams.get('dt_min', 5), # Get dt_min from hparams or use default
            'max_days': hparams.get('max_episode_days', 30) # Get max_days from hparams or use default
        }
        vec_env_cls = SubprocVecEnv if actual_num_cpu > 1 else DummyVecEnv
        env = make_vec_env(lambda: GreenhouseEnv(**env_kwargs), n_envs=actual_num_cpu, vec_env_cls=vec_env_cls)

        # Wrap with VecNormalize
        # Check if existing stats should be loaded (e.g., for continuing training)
        if os.path.exists(vec_norm_save_path):
             print(f"Loading existing VecNormalize stats from {vec_norm_save_path}")
             # When loading, pass the newly created (non-normalized) vec env
             env = VecNormalize.load(vec_norm_save_path, env)
             env.training = True # Make sure training mode is set
             print("Successfully loaded VecNormalize stats.")
        else:
             print("Creating new VecNormalize instance.")
             env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=hparams.get('gamma', 0.99))
        print(" Environment Created and Normalized.")


        # --- Callbacks ---
        checkpoint_callback = CheckpointCallback(
             # Save frequency needs to account for total steps across all envs
             save_freq=max(hparams.get('save_freq', 100000) // actual_num_cpu, 1),
             save_path=checkpoint_log_path,
             name_prefix=model_save_prefix,
             save_replay_buffer=False, # Not applicable to PPO
             save_vecnormalize=True # CRITICAL: Saves normalization stats with checkpoint
        )
        callback_list = [checkpoint_callback]

        # --- Define Model ---
        policy_kwargs = None
        if 'policy_kwargs' in hparams and isinstance(hparams['policy_kwargs'], str):
            try:
                # Use eval carefully only on your own controlled YAML files
                policy_kwargs = eval(hparams['policy_kwargs'])
                print(f" Parsed policy_kwargs: {policy_kwargs}")
            except Exception as e:
                print(f"WARN: Could not eval policy_kwargs string: {e}. Using default.")
        elif 'policy_kwargs' in hparams and isinstance(hparams['policy_kwargs'], dict):
             policy_kwargs = hparams['policy_kwargs'] # Use directly if already a dict

        # n_steps is per environment per update
        n_steps_per_env = hparams.get('n_steps', 2048) # SB3 recommends power of 2

        # Check if a checkpoint exists to continue training
        latest_checkpoint = None
        reset_num_timesteps = True # Default to starting fresh
        if os.path.exists(checkpoint_log_path):
            checkpoints = [f for f in os.listdir(checkpoint_log_path) if f.startswith(model_save_prefix) and f.endswith(".zip")]
            if checkpoints:
                try:
                    # Ensure correct parsing of step number, might need adjustment based on actual filename format
                    checkpoints.sort(key=lambda f: int(f.split('_')[-2].replace('steps','')))
                    latest_checkpoint = os.path.join(checkpoint_log_path, checkpoints[-1])
                    print(f"Found latest checkpoint: {latest_checkpoint}")
                except (IndexError, ValueError):
                     print(f"WARN: Could not parse step number from checkpoint filenames in {checkpoint_log_path}. Starting fresh.")


        if latest_checkpoint and os.path.exists(latest_checkpoint):
            print(f"Loading model from checkpoint: {latest_checkpoint}")
            # Pass the VecNormalized env instance
            model = PPO.load(latest_checkpoint, env=env, tensorboard_log=tensorboard_log_path)
            print("Model and VecNormalize stats loaded from checkpoint.")
            reset_num_timesteps = False # Continue timestep count
        else:
            if latest_checkpoint: print(f"WARN: Checkpoint {latest_checkpoint} not found, creating new model.")
            print("No valid checkpoint found, creating new PPO model.")
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=hparams.get('learning_rate', 3e-4),
                n_steps=n_steps_per_env,
                batch_size=hparams.get('batch_size', 64),
                n_epochs=hparams.get('n_epochs', 10),
                gamma=hparams.get('gamma', 0.99),
                gae_lambda=hparams.get('gae_lambda', 0.95),
                clip_range=hparams.get('clip_range', 0.2),
                ent_coef=hparams.get('ent_coef', 0.0),
                vf_coef=hparams.get('vf_coef', 0.5),
                max_grad_norm=hparams.get('max_grad_norm', 0.5),
                policy_kwargs=policy_kwargs,
                tensorboard_log=tensorboard_log_path,
                seed=hparams.get('seed', None),
                verbose=1
            )
        print(" PPO Model Initialized/Loaded.")

        # --- Train ---
        print(f"\n--- Starting/Resuming Training Loop ({total_timesteps} steps) ---")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            log_interval=10, # Print stats every 10 updates
            reset_num_timesteps=reset_num_timesteps
        )
        training_successful = True # Mark as successful if learn completes without error above

    except Exception as e:
         print(f"\nERROR during training setup or execution for {city} {mode}: {e}")
         traceback.print_exc()
         print("Attempting to save state before exiting...")
    finally:
        # --- Save Final Model & Stats ---
        print("\n--- Saving final model and normalization stats ---")
        try:
            # Only save if model object exists
            if model is not None:
                 model.save(model_save_path)
                 print(f" Final Model saved to: {model_save_path}.zip")
            else:
                 print(" Model object does not exist, cannot save model.")

            # Save VecNormalize stats regardless of training success, if env exists and is VecNormalize
            if env is not None and isinstance(env, VecNormalize):
                 env.save(vec_norm_save_path)
                 print(f" Final VecNormalize stats saved to: {vec_norm_save_path}")
            else:
                 print(" Could not save VecNormalize stats (env not VecNormalize instance or not created).")
        except Exception as e:
             print(f"ERROR saving final model/stats: {e}")

        end_time = time.time()
        print(f"--- Training Finished: {city.capitalize()} {mode.capitalize()} ---")
        print(f" Training Attempt Completed (Successful={training_successful})") # Indicate if learn() finished
        print(f" Total Time for this run: {(end_time - start_time) / 60:.2f} minutes")
        if env is not None:
            # Important: Close the VecEnv to terminate subprocesses if used
            env.close()
            print(" Environment closed.")

# --- End of function definition ---
print("Training function 'train_model' defined.")
```

    Training function 'train_model' defined.


## 4. Run Training Jobs

Execute the training function for each of the four models.

**Setup:**
- **Debug Run:** Set `FULL_TRAINING_RUN = False` in Cell 2. Run one of the cells below (e.g., Oslo Normal) to test the setup quickly (~1 minute). Check console output and TensorBoard.
- **Full Run:** Set `FULL_TRAINING_RUN = True` in Cell 2. Run all four cells below. This will take a significant amount of time (potentially hours).

**Monitoring:**
- Watch the console output for `ep_rew_mean` (should generally increase).
- Use TensorBoard in a separate terminal: `tensorboard --logdir logs/ppo_tensorboard` (run from project root).


```
# Cell 4a: Train Oslo Normal
# Ensure FULL_TRAINING_RUN is set appropriately in Cell 2 before running!
oslo_normal_trained_flag = False # Flag for diagnostics
if 'train_model' in locals() and callable(train_model):
    try:
        train_model(
            city='oslo',
            mode='normal',
            total_timesteps=DEFAULT_TOTAL_TIMESTEPS,
            num_cpu=DEFAULT_NUM_CPU
        )
        oslo_normal_trained_flag = True # Mark as attempted
    except Exception as e:
        print(f"FATAL ERROR during Oslo Normal training call setup: {e}")
        traceback.print_exc()
else:
    print("ERROR: train_model function not defined. Cannot train Oslo Normal.")
```

    
    --- Starting Training: Oslo Normal ---
     HParams File: /Users/jaabirahmed/Documents/AGARFARM/ml_training/config/oslo_normal_hparams.yaml
     City Config: /Users/jaabirahmed/Documents/AGARFARM/simulator/city_configs/oslo.yaml
     Model Save Prefix: /Users/jaabirahmed/Documents/AGARFARM/evaluation/trained_models/ppo_agent_o_normal
     VecNormalize Save Path: /Users/jaabirahmed/Documents/AGARFARM/evaluation/trained_models/vecnormalize_o_normal.pkl
     Tensorboard Logs: /Users/jaabirahmed/Documents/AGARFARM/logs/ppo_tensorboard/oslo_normal
     Checkpoints: /Users/jaabirahmed/Documents/AGARFARM/logs/ppo_checkpoints/oslo_normal
     Training for 10000 steps using 1 CPUs.
    GreenhouseEnv initialized: oslo.yaml, Mode: normal, Max Steps: 8640
    Loading existing VecNormalize stats from /Users/jaabirahmed/Documents/AGARFARM/evaluation/trained_models/vecnormalize_o_normal.pkl
    Successfully loaded VecNormalize stats.
     Environment Created and Normalized.
     Parsed policy_kwargs: {'net_arch': {'pi': [128, 128], 'vf': [128, 128]}}
    No valid checkpoint found, creating new PPO model.
    Using cpu device
     PPO Model Initialized/Loaded.
    
    --- Starting/Resuming Training Loop (10000 steps) ---
    Logging to /Users/jaabirahmed/Documents/AGARFARM/logs/ppo_tensorboard/oslo_normal/PPO_4
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 8.64e+03    |
    |    ep_rew_mean          | -2.61e+06   |
    | time/                   |             |
    |    fps                  | 2526        |
    |    iterations           | 10          |
    |    time_elapsed         | 4           |
    |    total_timesteps      | 10240       |
    | train/                  |             |
    |    approx_kl            | 0.019486334 |
    |    clip_fraction        | 0.214       |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -2.94       |
    |    explained_variance   | 0.549       |
    |    learning_rate        | 0.0003      |
    |    loss                 | 0.0286      |
    |    n_updates            | 90          |
    |    policy_gradient_loss | -0.0401     |
    |    value_loss           | 0.012       |
    -----------------------------------------
    
    --- Saving final model and normalization stats ---
     Final Model saved to: /Users/jaabirahmed/Documents/AGARFARM/evaluation/trained_models/ppo_agent_o_normal.zip
     Final VecNormalize stats saved to: /Users/jaabirahmed/Documents/AGARFARM/evaluation/trained_models/vecnormalize_o_normal.pkl
    --- Training Finished: Oslo Normal ---
     Training Attempt Completed (Successful=True)
     Total Time for this run: 0.07 minutes
     Environment closed.


# Cell 4b: Train Oslo Eco
# Ensure FULL_TRAINING_RUN is set appropriately in Cell 2 before running!


```
oslo_eco_trained_flag = False # Flag for diagnostics
if 'train_model' in locals() and callable(train_model):
    try:
        train_model(
            city='oslo',
            mode='eco',
            total_timesteps=DEFAULT_TOTAL_TIMESTEPS, # Or use different value from hparams if needed
            num_cpu=DEFAULT_NUM_CPU
        )
        oslo_eco_trained_flag = True
    except Exception as e:
        print(f"FATAL ERROR during Oslo Eco training call setup: {e}")
        traceback.print_exc()
else:
    print("ERROR: train_model function not defined. Cannot train Oslo Eco.")
```

    
    --- Starting Training: Oslo Eco ---
     HParams File: /Users/jaabirahmed/Documents/AGARFARM/ml_training/config/oslo_eco_hparams.yaml
     City Config: /Users/jaabirahmed/Documents/AGARFARM/simulator/city_configs/oslo.yaml
     Model Save Prefix: /Users/jaabirahmed/Documents/AGARFARM/evaluation/trained_models/ppo_agent_o_eco
     VecNormalize Save Path: /Users/jaabirahmed/Documents/AGARFARM/evaluation/trained_models/vecnormalize_o_eco.pkl
     Tensorboard Logs: /Users/jaabirahmed/Documents/AGARFARM/logs/ppo_tensorboard/oslo_eco
     Checkpoints: /Users/jaabirahmed/Documents/AGARFARM/logs/ppo_checkpoints/oslo_eco
     Training for 10000 steps using 1 CPUs.
    GreenhouseEnv initialized: oslo.yaml, Mode: eco, Max Steps: 8640
    Loading existing VecNormalize stats from /Users/jaabirahmed/Documents/AGARFARM/evaluation/trained_models/vecnormalize_o_eco.pkl
    Successfully loaded VecNormalize stats.
     Environment Created and Normalized.
     Parsed policy_kwargs: {'net_arch': {'pi': [128, 128], 'vf': [128, 128]}}
    No valid checkpoint found, creating new PPO model.
    Using cpu device
     PPO Model Initialized/Loaded.
    
    --- Starting/Resuming Training Loop (10000 steps) ---
    Logging to /Users/jaabirahmed/Documents/AGARFARM/logs/ppo_tensorboard/oslo_eco/PPO_2
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 8.64e+03    |
    |    ep_rew_mean          | -5.6e+05    |
    | time/                   |             |
    |    fps                  | 2595        |
    |    iterations           | 10          |
    |    time_elapsed         | 3           |
    |    total_timesteps      | 10240       |
    | train/                  |             |
    |    approx_kl            | 0.024129942 |
    |    clip_fraction        | 0.239       |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -2.91       |
    |    explained_variance   | 0.735       |
    |    learning_rate        | 0.0003      |
    |    loss                 | -0.0512     |
    |    n_updates            | 90          |
    |    policy_gradient_loss | -0.0445     |
    |    value_loss           | 0.0103      |
    -----------------------------------------
    
    --- Saving final model and normalization stats ---
     Final Model saved to: /Users/jaabirahmed/Documents/AGARFARM/evaluation/trained_models/ppo_agent_o_eco.zip
     Final VecNormalize stats saved to: /Users/jaabirahmed/Documents/AGARFARM/evaluation/trained_models/vecnormalize_o_eco.pkl
    --- Training Finished: Oslo Eco ---
     Training Attempt Completed (Successful=True)
     Total Time for this run: 0.07 minutes
     Environment closed.


# Cell 4c: Train Riyadh Normal
# Ensure FULL_TRAINING_RUN is set appropriately in Cell 2 before running!


```
riyadh_normal_trained_flag = False # Flag for diagnostics
if 'train_model' in locals() and callable(train_model):
    try:
        train_model(
            city='riyadh',
            mode='normal',
            total_timesteps=DEFAULT_TOTAL_TIMESTEPS,
            num_cpu=DEFAULT_NUM_CPU
        )
        riyadh_normal_trained_flag = True
    except Exception as e:
        print(f"FATAL ERROR during Riyadh Normal training call setup: {e}")
        traceback.print_exc()
else:
    print("ERROR: train_model function not defined. Cannot train Riyadh Normal.")
```

    
    --- Starting Training: Riyadh Normal ---
     HParams File: /Users/jaabirahmed/Documents/AGARFARM/ml_training/config/riyadh_normal_hparams.yaml
     City Config: /Users/jaabirahmed/Documents/AGARFARM/simulator/city_configs/riyadh.yaml
     Model Save Prefix: /Users/jaabirahmed/Documents/AGARFARM/evaluation/trained_models/ppo_agent_r_normal
     VecNormalize Save Path: /Users/jaabirahmed/Documents/AGARFARM/evaluation/trained_models/vecnormalize_r_normal.pkl
     Tensorboard Logs: /Users/jaabirahmed/Documents/AGARFARM/logs/ppo_tensorboard/riyadh_normal
     Checkpoints: /Users/jaabirahmed/Documents/AGARFARM/logs/ppo_checkpoints/riyadh_normal
     Training for 10000 steps using 1 CPUs.
    GreenhouseEnv initialized: riyadh.yaml, Mode: normal, Max Steps: 8640
    Loading existing VecNormalize stats from /Users/jaabirahmed/Documents/AGARFARM/evaluation/trained_models/vecnormalize_r_normal.pkl
    Successfully loaded VecNormalize stats.
     Environment Created and Normalized.
     Parsed policy_kwargs: {'net_arch': {'pi': [128, 128], 'vf': [128, 128]}}
    No valid checkpoint found, creating new PPO model.
    Using cpu device
     PPO Model Initialized/Loaded.
    
    --- Starting/Resuming Training Loop (10000 steps) ---
    Logging to /Users/jaabirahmed/Documents/AGARFARM/logs/ppo_tensorboard/riyadh_normal/PPO_4
    ----------------------------------------
    | rollout/                |            |
    |    ep_len_mean          | 8.64e+03   |
    |    ep_rew_mean          | -4.29e+04  |
    | time/                   |            |
    |    fps                  | 2576       |
    |    iterations           | 10         |
    |    time_elapsed         | 3          |
    |    total_timesteps      | 10240      |
    | train/                  |            |
    |    approx_kl            | 0.02360581 |
    |    clip_fraction        | 0.195      |
    |    clip_range           | 0.2        |
    |    entropy_loss         | -2.92      |
    |    explained_variance   | 0.247      |
    |    learning_rate        | 0.0003     |
    |    loss                 | -0.0697    |
    |    n_updates            | 90         |
    |    policy_gradient_loss | -0.0406    |
    |    value_loss           | 0.00544    |
    ----------------------------------------
    
    --- Saving final model and normalization stats ---
     Final Model saved to: /Users/jaabirahmed/Documents/AGARFARM/evaluation/trained_models/ppo_agent_r_normal.zip
     Final VecNormalize stats saved to: /Users/jaabirahmed/Documents/AGARFARM/evaluation/trained_models/vecnormalize_r_normal.pkl
    --- Training Finished: Riyadh Normal ---
     Training Attempt Completed (Successful=True)
     Total Time for this run: 0.07 minutes
     Environment closed.


# Cell 4d: Train Riyadh Eco
# Ensure FULL_TRAINING_RUN is set appropriately in Cell 2 before running!


```
riyadh_eco_trained_flag = False # Flag for diagnostics
if 'train_model' in locals() and callable(train_model):
    try:
        train_model(
            city='riyadh',
            mode='eco',
            total_timesteps=DEFAULT_TOTAL_TIMESTEPS, # Or use different value from hparams if needed
            num_cpu=DEFAULT_NUM_CPU
        )
        riyadh_eco_trained_flag = True
    except Exception as e:
        print(f"FATAL ERROR during Riyadh Eco training call setup: {e}")
        traceback.print_exc()
else:
    print("ERROR: train_model function not defined. Cannot train Riyadh Eco.")
```

    
    --- Starting Training: Riyadh Eco ---
     HParams File: /Users/jaabirahmed/Documents/AGARFARM/ml_training/config/riyadh_eco_hparams.yaml
     City Config: /Users/jaabirahmed/Documents/AGARFARM/simulator/city_configs/riyadh.yaml
     Model Save Prefix: /Users/jaabirahmed/Documents/AGARFARM/evaluation/trained_models/ppo_agent_r_eco
     VecNormalize Save Path: /Users/jaabirahmed/Documents/AGARFARM/evaluation/trained_models/vecnormalize_r_eco.pkl
     Tensorboard Logs: /Users/jaabirahmed/Documents/AGARFARM/logs/ppo_tensorboard/riyadh_eco
     Checkpoints: /Users/jaabirahmed/Documents/AGARFARM/logs/ppo_checkpoints/riyadh_eco
     Training for 10000 steps using 1 CPUs.
    GreenhouseEnv initialized: riyadh.yaml, Mode: eco, Max Steps: 8640
    Loading existing VecNormalize stats from /Users/jaabirahmed/Documents/AGARFARM/evaluation/trained_models/vecnormalize_r_eco.pkl
    Successfully loaded VecNormalize stats.
     Environment Created and Normalized.
     Parsed policy_kwargs: {'net_arch': {'pi': [128, 128], 'vf': [128, 128]}}
    No valid checkpoint found, creating new PPO model.
    Using cpu device
     PPO Model Initialized/Loaded.
    
    --- Starting/Resuming Training Loop (10000 steps) ---
    Logging to /Users/jaabirahmed/Documents/AGARFARM/logs/ppo_tensorboard/riyadh_eco/PPO_2
    -----------------------------------------
    | rollout/                |             |
    |    ep_len_mean          | 8.64e+03    |
    |    ep_rew_mean          | -3.55e+03   |
    | time/                   |             |
    |    fps                  | 2545        |
    |    iterations           | 10          |
    |    time_elapsed         | 4           |
    |    total_timesteps      | 10240       |
    | train/                  |             |
    |    approx_kl            | 0.014446383 |
    |    clip_fraction        | 0.16        |
    |    clip_range           | 0.2         |
    |    entropy_loss         | -3.01       |
    |    explained_variance   | 0.386       |
    |    learning_rate        | 0.0003      |
    |    loss                 | -0.0271     |
    |    n_updates            | 90          |
    |    policy_gradient_loss | -0.0362     |
    |    value_loss           | 0.0355      |
    -----------------------------------------
    
    --- Saving final model and normalization stats ---
     Final Model saved to: /Users/jaabirahmed/Documents/AGARFARM/evaluation/trained_models/ppo_agent_r_eco.zip
     Final VecNormalize stats saved to: /Users/jaabirahmed/Documents/AGARFARM/evaluation/trained_models/vecnormalize_r_eco.pkl
    --- Training Finished: Riyadh Eco ---
     Training Attempt Completed (Successful=True)
     Total Time for this run: 0.07 minutes
     Environment closed.


## 5. Conclusion & Diagnostics

Review the training output and check for saved files using the diagnostics cell below.


```
# Cell 5: Diagnostics Summary for Review
# --- Notebook 03 Diagnostics Cell ---
import pandas as pd
import numpy as np
from datetime import datetime
import os
import traceback

print("--- Notebook 03 Diagnostics ---")
print(f"Timestamp: {datetime.now()}")

# Define expected output directories and file patterns
# Ensure MODEL_SAVE_DIR is defined in a previous cell, or redefine it here:
if 'MODEL_SAVE_DIR' not in locals():
    try:
        notebook_dir = os.getcwd()
        project_root = os.path.abspath(os.path.join(notebook_dir, '..'))
        MODEL_SAVE_DIR = os.path.abspath(os.path.join(project_root, "evaluation", "trained_models"))
        print(f"INFO: Redefined MODEL_SAVE_DIR to {MODEL_SAVE_DIR}")
    except Exception as path_e:
        print(f"ERROR: Could not define MODEL_SAVE_DIR: {path_e}")
        MODEL_SAVE_DIR = "./" # Fallback


cases = [('oslo', 'normal'), ('oslo', 'eco'), ('riyadh', 'normal'), ('riyadh', 'eco')]
all_files_found_dict = {} # Track found status per case
any_missing = False

# Check if train_model function exists (as a basic check it was defined)
train_func_defined = 'train_model' in locals() and callable(train_model)
print(f"\nTraining Function Defined: {train_func_defined}")

# Check for output files
print("\nChecking for Saved Model & Stats Files:")
if not os.path.isdir(MODEL_SAVE_DIR):
     print(f"ERROR: Model save directory does not exist: {MODEL_SAVE_DIR}")
     any_missing = True # Mark all as missing if dir doesn't exist
else:
    for city, mode in cases:
        city_code = city[0]
        model_suffix = f"{city_code}_{mode}"
        model_filename = f"ppo_agent_{model_suffix}.zip"
        vec_norm_filename = f"vecnormalize_{model_suffix}.pkl"
        model_filepath = os.path.join(MODEL_SAVE_DIR, model_filename)
        vec_norm_filepath = os.path.join(MODEL_SAVE_DIR, vec_norm_filename) # Assuming saved in same dir

        model_found = os.path.exists(model_filepath)
        stats_found = os.path.exists(vec_norm_filepath)
        all_files_found_dict[f"{city}_{mode}"] = model_found and stats_found
        print(f"- {city.capitalize()} {mode.capitalize()}:")
        print(f"  - Model (.zip): {'Found' if model_found else 'MISSING'} ({model_filename})")
        print(f"  - Stats (.pkl): {'Found' if stats_found else 'MISSING'} ({vec_norm_filename})")
        if not model_found or not stats_found:
            any_missing = True

if not any_missing:
    print("\nStatus: All expected model and stats files appear to be present.")
else:
    print("\nWARNING: Some model or stats files are missing. Training may not have completed successfully for all cases, or files were saved elsewhere.")

# Check flags if they were set in the calling cells
print("\nTraining Runs Attempted (based on notebook cell execution flags):")
print(f"- Oslo Normal Attempted: {'oslo_normal_trained_flag' in locals() and oslo_normal_trained_flag}")
print(f"- Oslo Eco Attempted: {'oslo_eco_trained_flag' in locals() and oslo_eco_trained_flag}")
print(f"- Riyadh Normal Attempted: {'riyadh_normal_trained_flag' in locals() and riyadh_normal_trained_flag}")
print(f"- Riyadh Eco Attempted: {'riyadh_eco_trained_flag' in locals() and riyadh_eco_trained_flag}")


print("\n--- End Notebook 03 Diagnostics ---")
```

    --- Notebook 03 Diagnostics ---
    Timestamp: 2025-05-06 20:09:22.748813
    
    Training Function Defined: True
    
    Checking for Saved Model & Stats Files:
    - Oslo Normal:
      - Model (.zip): Found (ppo_agent_o_normal.zip)
      - Stats (.pkl): Found (vecnormalize_o_normal.pkl)
    - Oslo Eco:
      - Model (.zip): Found (ppo_agent_o_eco.zip)
      - Stats (.pkl): Found (vecnormalize_o_eco.pkl)
    - Riyadh Normal:
      - Model (.zip): Found (ppo_agent_r_normal.zip)
      - Stats (.pkl): Found (vecnormalize_r_normal.pkl)
    - Riyadh Eco:
      - Model (.zip): Found (ppo_agent_r_eco.zip)
      - Stats (.pkl): Found (vecnormalize_r_eco.pkl)
    
    Status: All expected model and stats files appear to be present.
    
    Training Runs Attempted (based on notebook cell execution flags):
    - Oslo Normal Attempted: True
    - Oslo Eco Attempted: True
    - Riyadh Normal Attempted: True
    - Riyadh Eco Attempted: True
    
    --- End Notebook 03 Diagnostics ---

