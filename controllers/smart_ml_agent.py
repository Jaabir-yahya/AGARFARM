import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Any, Optional, Tuple

# Stable Baselines3 and Gymnasium
try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    print("INFO: Gymnasium and Stable-Baselines3 (PPO) imported successfully in smart_ml_agent.")
except ImportError:
    print("WARN: stable-baselines3 or gymnasium not installed. SmartMLAgent cannot load SB3 models.")
    PPO = None; VecNormalize = None; spaces = None; MinimalMatchingEnv = None

# Simulator type hinting
try:
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    if project_root not in sys.path: sys.path.insert(0, project_root)
    from simulator.core import ControlState
except ImportError:
    ControlState = Any
    print("WARN: Could not import ControlState for type hinting in smart_ml_agent.")

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Minimal Dummy Environment with Matching 28-Feature Space ---
if gym is not None and spaces is not None:
    class MinimalMatchingEnv(gym.Env):
        """Dummy Env for loading VecNormalize stats with the 28-feature space."""
        def __init__(self, obs_dim=28, action_dim=24): # Use parameters for flexibility
            super().__init__()
            self.obs_dim = obs_dim
            self.action_dim = action_dim

            # Define bounds roughly - exact bounds less critical for VecNormalize loading
            if self.obs_dim == 28:
                OBS_LOW_BASE  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -20.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], dtype=np.float32)
                OBS_HIGH_BASE = np.array([50.0, 100.0, 100.0, 5.0, 500.0, 50.0, 100.0, 1.0, 1.0, 1.0, 50.0, 50.0, 100.0, 100.0, 100.0, 100.0 ], dtype=np.float32)
                OBS_LOW_HIST = np.zeros(12, dtype=np.float32)
                OBS_HIGH_HIST = np.ones(12, dtype=np.float32)
                OBS_LOW_28 = np.concatenate((OBS_LOW_BASE, OBS_LOW_HIST))
                OBS_HIGH_28 = np.concatenate((OBS_HIGH_BASE, OBS_HIGH_HIST))
                low_bounds = OBS_LOW_28
                high_bounds = OBS_HIGH_28
            else: # Fallback for potential errors or different phases
                 logger.warning(f"MinimalMatchingEnv initialized with unexpected obs_dim {self.obs_dim}. Using generic bounds.")
                 low_bounds = -np.inf * np.ones(self.obs_dim, dtype=np.float32)
                 high_bounds = np.inf * np.ones(self.obs_dim, dtype=np.float32)

            self.observation_space = spaces.Box(
                low=low_bounds, high=high_bounds,
                shape=(self.obs_dim,), dtype=np.float32 )
            self.action_space = spaces.Discrete(self.action_dim)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}
        def step(self, action):
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}
        def render(self, mode='human'): pass
        def close(self): pass
else:
    MinimalMatchingEnv = None


class SmartMLAgent:
    """Loads SB3 PPO model & VecNormalize stats, predicts actions."""
    def __init__(self, model_path: str, vec_normalize_path: Optional[str] = None):
        self.model: Optional[PPO] = None
        self.vec_normalize: Optional[VecNormalize] = None
        self.model_path = model_path
        self.model_name = os.path.basename(model_path) if model_path else None
        self.expected_obs_shape = (28,) # Expected shape for Phase 3B models

        if PPO is None or VecNormalize is None or MinimalMatchingEnv is None:
            logger.error("SB3/Gymnasium or MinimalMatchingEnv not available. SmartMLAgent unusable.")
            self.model_metadata = self._extract_model_metadata()
            return

        try:
            self.model = PPO.load(model_path, device='cpu')
            logger.info(f"ML Model loaded: {model_path}")

            # Verify model observation space if possible
            model_shape_info = "N/A (No model loaded)"
            if hasattr(self.model, 'observation_space') and hasattr(self.model.observation_space, 'shape'):
                model_shape = self.model.observation_space.shape
                model_shape_info = str(model_shape)
                if model_shape != self.expected_obs_shape:
                    logger.warning(f"Loaded model expects obs shape {model_shape}, but Agent configured for {self.expected_obs_shape}.")
            else:
                 logger.warning("Could not verify observation shape from loaded model.")


            if vec_normalize_path and os.path.exists(vec_normalize_path):
                logger.info(f"Loading VecNormalize: {vec_normalize_path}...")
                # Use obs_dim from expected shape for dummy env
                dummy_env = DummyVecEnv([lambda: MinimalMatchingEnv(obs_dim=self.expected_obs_shape[0])])
                try:
                    self.vec_normalize = VecNormalize.load(vec_normalize_path, dummy_env)
                    self.vec_normalize.training = False; self.vec_normalize.norm_reward = False
                    logger.info(f"VecNormalize loaded: {vec_normalize_path}")
                    # Verify VecNormalize stats match expected shape
                    if hasattr(self.vec_normalize, 'observation_space') and \
                       self.vec_normalize.observation_space.shape != self.expected_obs_shape:
                         logger.warning(f"Loaded VecNormalize obs space shape {self.vec_normalize.observation_space.shape} != expected {self.expected_obs_shape}.")
                except Exception as e:
                    logger.error(f"Failed loading VecNormalize: {e}. Proceeding without.", exc_info=True)
                    self.vec_normalize = None
                finally:
                    dummy_env.close()
            elif vec_normalize_path: logger.warning(f"VecNormalize path specified but not found: {vec_normalize_path}.")
            else: logger.info("No VecNormalize path specified.")

        except Exception as e:
            logger.error(f"Could not load PPO model: {e}", exc_info=True)
            self.model = None

        # Extract metadata (includes model_shape_info now)
        self.model_metadata = self._extract_model_metadata(model_shape_info)


    def _extract_model_metadata(self, model_shape_info="N/A") -> dict:
        return {
            "algo": "PPO", "framework": "stable-baselines3", "source": self.model_path,
            "model_name": self.model_name, "model_loaded": self.model is not None,
            "model_observation_space": model_shape_info, "trained_on": "unknown",
            "vec_normalize_loaded": self.vec_normalize is not None,
            "vec_normalize_source": getattr(self.vec_normalize, 'load_path', None) if self.vec_normalize else None }

    def _map_discrete_action_to_controlstate(self, discrete_action: int) -> ControlState:
        action_space_size = 24
        if not 0 <= discrete_action < action_space_size:
            logger.warning(f"Invalid action {discrete_action}, defaulting to OFF.")
            return ControlState()
        level = discrete_action // 8; part = discrete_action % 8
        return ControlState( fan=bool(part & 4), ac=(level > 0), vent=bool(part & 2), irrigation=bool(part & 1) )

    def get_action(self, observation: np.ndarray) -> ControlState:
        if self.model is None:
            logger.warning("No model loaded, returning default ControlState.")
            return ControlState()

        if observation.shape[-1] != self.expected_obs_shape[-1]:
             logger.error(f"Obs dimension mismatch! Expected {self.expected_obs_shape[-1]}, got {observation.shape[-1]}.")
             return ControlState()

        try:
            obs_reshaped = observation.reshape(1, -1)
            obs_normalized = obs_reshaped
            if self.vec_normalize is not None:
                if hasattr(self.vec_normalize, 'observation_space') and \
                   self.vec_normalize.observation_space.shape == self.expected_obs_shape:
                     obs_normalized = self.vec_normalize.normalize_obs(obs_reshaped)
                else:
                     logger.warning(f"VecNormalize obs space shape mismatch. Skipping normalization.")

            action_discrete, _ = self.model.predict(obs_normalized, deterministic=True)
            return self._map_discrete_action_to_controlstate(action_discrete[0])

        except Exception as e:
            logger.error(f"ERROR during ML prediction: {e}", exc_info=True)
            return ControlState()