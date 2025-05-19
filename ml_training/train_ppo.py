import argparse
import yaml
import os
import time
import gymnasium as gym  # Preferred over gym for latest SB3
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
# from stable_baselines3.common.monitor import Monitor # Monitor is auto-wrapped by make_vec_env
import traceback
import logging  # For better logging

# --- Project Path Setup ---
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Module Imports ---
try:
    from ml_training.custom_env_wrapper import GreenhouseEnv

    # Import any constants needed for default user_simulator_params if not hardcoded
    # from simulator.core import DEFAULT_SIMULATOR_MULTIPLIERS
    logging.info("Imported custom GreenhouseEnv wrapper successfully for training.")
except ImportError as e:
    logging.error(f"ERROR: custom_env_wrapper.py not found or GreenhouseEnv class missing: {e}", exc_info=True)
    raise  # Critical, cannot proceed

# --- Define Constants (can be overridden by hparams) ---
DEFAULT_DT_MIN_TRAIN = 5
DEFAULT_MAX_DAYS_TRAIN = 30  # Default episode length for training

# --- Default User Simulator Parameters for Training ---
# These ensure training happens on a 'standard' simulator unless overridden by a specific research goal
DEFAULT_USER_SIMULATOR_PARAMS_FOR_TRAINING = {
    "rain_intensity_multiplier": 1.0,
    "rain_probability_multiplier": 1.0,
    "plant_transpiration_multiplier": 1.0,
    "soil_drying_multiplier": 1.0
}


def train(city: str, mode: str, config_path: str, model_save_dir: str, vec_norm_save_dir: str):
    """Trains a PPO agent using Stable Baselines3, ALWAYS STARTING NEW."""
    overall_start_time = time.time()
    logger = logging.getLogger(f"TrainPPO.{city}.{mode}")  # Specific logger
    logger.info(f"--- Starting OFFLINE Training (ALWAYS NEW): {city.capitalize()} {mode.capitalize()} ---")
    logger.info(f"Config: {config_path}")

    # --- Define Paths ---
    model_save_prefix = f"ppo_agent_{city[0]}_{mode}"  # e.g. ppo_agent_o_normal
    model_save_path = os.path.join(model_save_dir, model_save_prefix)
    vec_norm_save_path = os.path.join(vec_norm_save_dir, f"vecnormalize_{city[0]}_{mode}.pkl")
    tensorboard_log_path = os.path.join(project_root, "logs", "ppo_tensorboard", f"{city}_{mode}")
    checkpoint_log_path = os.path.join(project_root, "logs", "ppo_checkpoints", f"{city}_{mode}")

    os.makedirs(tensorboard_log_path, exist_ok=True)
    os.makedirs(checkpoint_log_path, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(vec_norm_save_dir, exist_ok=True)

    logger.info(f"  Model Save Path: {model_save_path}.zip")
    logger.info(f"  VecNormalize Save Path: {vec_norm_save_path}")
    logger.info(f"  Tensorboard Logs: {tensorboard_log_path}")
    logger.info(f"  Checkpoints: {checkpoint_log_path}")

    # --- 1. Load Configs ---
    city_config_file = os.path.join(project_root, "simulator", "city_configs", f"{city}.yaml")
    if not os.path.exists(city_config_file):
        logger.error(f"City config not found: {city_config_file}");
        raise FileNotFoundError(f"City config not found: {city_config_file}")
    if not os.path.exists(config_path):
        logger.error(f"Hyperparameter config not found: {config_path}");
        raise FileNotFoundError(f"Hyperparameter config not found: {config_path}")

    with open(config_path, 'r') as f:
        hparams = yaml.safe_load(f)
    logger.info(f"  Loaded Hyperparameters: {hparams}")
    with open(city_config_file, 'r') as f:
        city_cfg_data = yaml.safe_load(f)

    target_ranges = {k.replace('default_target_', ''): v for k, v in city_cfg_data.items() if
                     k.startswith('default_target_')}
    logger.info(f"  Using Target Ranges: {target_ranges}")
    reward_params = hparams.get('reward_params')
    if reward_params is None:
        logger.error(f"Reward parameters ('reward_params') not found in {config_path}");
        raise ValueError("reward_params not found")
    logger.info(f"  Using Reward Parameters: {reward_params}")

    # --- 2. Create Vectorized Environment ---
    actual_num_cpu = max(1, min(hparams.get('num_cpu', 1), os.cpu_count() or 1))  # os.cpu_count() can be None
    logger.info(f"  Creating VecEnv with {actual_num_cpu} parallel environment(s)...")

    env_kwargs = {
        'city_config_path': city_config_file,
        'target_ranges': target_ranges,
        'mode': mode,
        'reward_params': reward_params,
        'dt_min': hparams.get('dt_min', DEFAULT_DT_MIN_TRAIN),
        'max_days': hparams.get('max_episode_days', DEFAULT_MAX_DAYS_TRAIN),
        'user_simulator_params': DEFAULT_USER_SIMULATOR_PARAMS_FOR_TRAINING  # <-- Pass default sim params
    }
    vec_env_cls = SubprocVecEnv if actual_num_cpu > 1 else DummyVecEnv
    logger.info(f"  Using {vec_env_cls.__name__} for VecEnv.")

    # make_vec_env automatically wraps each env with Monitor
    env = make_vec_env(lambda: GreenhouseEnv(**env_kwargs), n_envs=actual_num_cpu, vec_env_cls=vec_env_cls)

    if os.path.exists(vec_norm_save_path):
        logger.info(f"  INFO: Old VecNormalize stats file found at {vec_norm_save_path}. It will be overwritten.")

    logger.info("  Creating new VecNormalize instance for fresh training.")
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=hparams.get('gamma', 0.99))
    logger.info("  Environment Created and Normalized.")

    # --- 3. Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=max(hparams.get('save_freq', 100000) // actual_num_cpu, 1),
        save_path=checkpoint_log_path, name_prefix=model_save_prefix,
        save_replay_buffer=False, save_vecnormalize=True)
    callback_list = [checkpoint_callback]

    # --- 4. Define Model ---
    policy_kwargs_from_yaml = hparams.get('policy_kwargs', {})  # Default to empty dict
    if isinstance(policy_kwargs_from_yaml, str):
        try:
            policy_kwargs = eval(policy_kwargs_from_yaml)
        except Exception as e:
            logger.warning(f"Could not eval policy_kwargs: {e}. Using default."); policy_kwargs = {}
    else:
        policy_kwargs = policy_kwargs_from_yaml

    logger.info("  Creating new PPO model (ignoring any existing checkpoints).")
    model = PPO(
        "MlpPolicy", env,
        learning_rate=hparams.get('learning_rate', 3e-4),
        n_steps=hparams.get('n_steps', 2048),
        batch_size=hparams.get('batch_size', 64),
        n_epochs=hparams.get('n_epochs', 10),
        gamma=hparams.get('gamma', 0.99),
        gae_lambda=hparams.get('gae_lambda', 0.95),
        clip_range=hparams.get('clip_range', 0.2),
        ent_coef=hparams.get('ent_coef', 0.01),  # Tuned this from 0.0
        vf_coef=hparams.get('vf_coef', 0.5),
        max_grad_norm=hparams.get('max_grad_norm', 0.5),
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log_path,
        seed=hparams.get('seed', None),
        verbose=1)
    logger.info("  PPO Model Initialized.")

    # --- 5. Train ---
    training_successful_flag = False
    target_total_timesteps = hparams.get('training_timesteps', 1_000_000)
    logger.info(f"  Starting NEW PPO training for {target_total_timesteps} timesteps.")
    actual_learn_time_start = time.time()
    try:
        model.learn(
            total_timesteps=target_total_timesteps,
            callback=callback_list,
            log_interval=10,  # How often to print SB3 logs
            reset_num_timesteps=True)  # Crucial for new training
        training_successful_flag = True
    except Exception as e:
        logger.error(f"ERROR during training: {e}", exc_info=True)
        # traceback.print_exc() # Already logged with exc_info=True

    actual_learn_time_end = time.time()
    logger.info(
        f"  Actual training/learn() call duration: {(actual_learn_time_end - actual_learn_time_start) / 60:.2f} minutes")

    # --- 6. Save Final Model & Stats ---
    logger.info("--- Saving final model and normalization stats ---")
    try:
        if model is not None:  # Should exist unless PPO init failed
            model.save(model_save_path)
            logger.info(f" Final Model saved to: {model_save_path}.zip")
        if env is not None and isinstance(env, VecNormalize):
            env.save(vec_norm_save_path)
            logger.info(f" Final VecNormalize stats saved to: {vec_norm_save_path}")
        else:
            logger.warning("Environment is None or not VecNormalize, cannot save stats.")
    except Exception as e:
        logger.error(f"ERROR saving final model/stats: {e}", exc_info=True)

    overall_end_time = time.time()
    logger.info(f"--- OFFLINE Training Function Completed: {city.capitalize()} {mode.capitalize()} ---")
    logger.info(f" Training Attempt Overall (Successful={training_successful_flag})")
    logger.info(
        f" Total Time for this train() function call: {(overall_end_time - overall_start_time) / 60:.2f} minutes")
    if env is not None:
        try:
            env.close()
        except Exception as e:
            logger.error(f"Error closing environment: {e}", exc_info=True)


if __name__ == "__main__":
    # Setup basic logging for the main script execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Offline Train PPO Controller (Always New Training)")
    parser.add_argument("--city", required=True, choices=['oslo', 'riyadh'], help="City")
    parser.add_argument("--mode", required=True, choices=['normal', 'eco'], help="Mode")
    args = parser.parse_args()

    hparam_config_file = os.path.join(current_dir, "config", f"{args.city}_{args.mode}_hparams.yaml")
    save_dir_models = os.path.join(project_root, "evaluation", "trained_models")
    # VecNormalize stats are usually saved in the same directory as models by convention

    try:
        # A quick check if GreenhouseEnv class is usable before starting
        if not hasattr(GreenhouseEnv, 'step'):  # Basic check
            logging.critical("GreenhouseEnv does not seem to be correctly defined/imported. Exiting.")
        else:
            train(args.city, args.mode, hparam_config_file, save_dir_models, save_dir_models)
    except Exception as e:
        logging.critical(f"Unhandled exception in __main__: {e}", exc_info=True)
