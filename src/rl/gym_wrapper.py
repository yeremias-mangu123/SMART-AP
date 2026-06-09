"""Gymnasium wrappers for single and multi-layout AP placement training."""

import copy
import random
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np

from .environment import WiFiAPPlacementEnv


class GymWiFiAPPlacementEnv(gym.Env):
    """Gymnasium environment for AP Placement using PPO."""

    candidate_feature_dim = 9
    
    def __init__(self, scenario: Dict[str, Any]):
        super().__init__()
        self.env = WiFiAPPlacementEnv(scenario)
        
        # Action space: [candidate_index, ap_model_index, power_index, stop_flag]
        shape = self.env.action_space_shape
        # We need to make sure shape dimensions are > 0.
        # If there are no candidates, we create a dummy space to avoid crashes.
        action_dims = np.array([max(1, dim) for dim in shape])
        self.action_space = gym.spaces.MultiDiscrete(action_dims)
        
        # Observation space. PPO MultiInputPolicy requires a Dict space; scalar
        # metrics describe progress, while candidate tensors let the policy
        # associate candidate_index actions with actual layout features.
        obs = self.env.get_observation()
        spaces = {}
        for key, value in obs.items():
            if isinstance(value, float):
                spaces[key] = gym.spaces.Box(low=-1e6, high=1e6, shape=(1,), dtype=np.float32)
            elif isinstance(value, int):
                spaces[key] = gym.spaces.Box(low=-1e6, high=1e6, shape=(1,), dtype=np.float32)

        candidate_count = int(action_dims[0])
        spaces["candidate_features"] = gym.spaces.Box(
            low=-1e6,
            high=1e6,
            shape=(candidate_count * self.candidate_feature_dim,),
            dtype=np.float32,
        )
        spaces["candidate_valid_mask"] = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(candidate_count,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(spaces)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset()
        return self._format_obs(obs), info

    def step(self, action):
        # Action is a numpy array from MultiDiscrete, convert to list
        action_list = action.tolist() if hasattr(action, "tolist") else list(action)
        # Convert stop flag to boolean just in case
        if len(action_list) > 3:
            action_list[3] = bool(action_list[3])
            
        obs, reward, terminated, truncated, info = self.env.step(action_list)
        return self._format_obs(obs), reward, terminated, truncated, info

    def _format_obs(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Convert environment observations into arrays expected by SB3."""
        formatted = {}
        for key in self.observation_space.spaces:
            if key in {"candidate_features", "candidate_valid_mask"}:
                continue
            formatted[key] = np.array([obs.get(key, 0.0)], dtype=np.float32)

        candidate_count = int(self.action_space.nvec[0])
        features = np.zeros((candidate_count, self.candidate_feature_dim), dtype=np.float32)
        raw_features = np.array(self.env.candidate_features(), dtype=np.float32)
        if raw_features.size:
            rows = min(candidate_count, raw_features.shape[0])
            features[:rows, : raw_features.shape[1]] = raw_features[:rows, : self.candidate_feature_dim]
        formatted["candidate_features"] = features.reshape(-1)

        valid_mask = np.zeros((candidate_count,), dtype=np.float32)
        raw_mask = self.env.candidate_action_mask()
        if raw_mask:
            rows = min(candidate_count, len(raw_mask))
            valid_mask[:rows] = np.array(raw_mask[:rows], dtype=np.float32)
        elif candidate_count:
            valid_mask[0] = 1.0
        formatted["candidate_valid_mask"] = valid_mask
        return formatted

    def action_masks(self) -> np.ndarray:
        """Return flattened MultiDiscrete action masks for sb3-contrib.

        The order follows the MultiDiscrete dimensions:
        candidate, AP model, power level, stop flag.
        """
        candidate_count, model_count, power_count, _ = [int(value) for value in self.action_space.nvec]

        candidate_mask = list(self.env.candidate_action_mask())
        if not candidate_mask:
            candidate_mask = [True]
        if not any(candidate_mask):
            candidate_mask = [True] + [False] * (candidate_count - 1)
        candidate_mask = (candidate_mask + [False] * candidate_count)[:candidate_count]

        band = "6" if self.env.frequency_ghz >= 6 else ("5" if self.env.frequency_ghz >= 5 else "2.4")
        model_mask = []
        for model in self.env.ap_models:
            model_mask.append(band in [str(item) for item in model.get("bands", ["2.4", "5"])])
        if not model_mask:
            model_mask = [True]
        if not any(model_mask):
            model_mask[0] = True
        model_mask = (model_mask + [False] * model_count)[:model_count]

        power_mask = [True] * power_count
        stop_mask = [True, bool(self.env.can_stop())]
        return np.array(candidate_mask + model_mask + power_mask + stop_mask, dtype=bool)


class MultiScenarioGymWiFiAPPlacementEnv(GymWiFiAPPlacementEnv):
    """Sample a different compatible floor plan at every episode reset."""

    def __init__(self, scenarios: List[Dict[str, Any]], seed: int = 42):
        if not scenarios:
            raise ValueError("At least one scenario is required")
        self.scenarios = [copy.deepcopy(scenario) for scenario in scenarios]
        self.rng = random.Random(seed)
        super().__init__(self.scenarios[0])
        expected_action_shape = tuple(int(value) for value in self.action_space.nvec)
        expected_observation = self.observation_space
        for scenario in self.scenarios[1:]:
            probe = GymWiFiAPPlacementEnv(scenario)
            if tuple(int(value) for value in probe.action_space.nvec) != expected_action_shape:
                raise ValueError("Multi-scenario training requires identical action-space dimensions")
            if probe.observation_space != expected_observation:
                raise ValueError("Multi-scenario training requires identical observation spaces")

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng.seed(seed)
        scenario = self.rng.choice(self.scenarios)
        self.env = WiFiAPPlacementEnv(copy.deepcopy(scenario))
        return super().reset(seed=seed, options=options)
