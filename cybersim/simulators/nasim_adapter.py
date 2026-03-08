"""NASim adapter implementing SimulatorInterface."""

import gymnasium as gym
import nasim
import numpy as np
import yaml

from cybersim.simulators.base import SimulatorInterface


class NaSimAdapter(SimulatorInterface):
    """Wraps NASim behind the SimulatorInterface."""

    def __init__(self, config_path: str) -> None:
        self._config_path = config_path
        self._sensitive_hosts = self._parse_sensitive_hosts()
        self._env = nasim.load(
            config_path,
            fully_obs=True,
            flat_actions=True,
            flat_obs=True,
        )

    def _parse_sensitive_hosts(self) -> list[tuple[int, int]]:
        with open(self._config_path) as f:
            config = yaml.safe_load(f)
        raw = config.get("sensitive_hosts", {})
        hosts = []
        for key in raw:
            if isinstance(key, str):
                cleaned = key.strip("() ")
                parts = [int(x.strip()) for x in cleaned.split(",")]
                hosts.append(tuple(parts))
            elif isinstance(key, tuple):
                hosts.append(key)
        return hosts

    def reset(self) -> np.ndarray:
        result = self._env.reset()
        if isinstance(result, tuple):
            return result[0]
        return result

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        result = self._env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, float(reward), terminated or truncated, info
        obs, reward, done, info = result
        return obs, float(reward), done, info

    def get_action_space(self) -> gym.Space:
        return self._env.action_space

    def get_observation_space(self) -> gym.Space:
        return self._env.observation_space

    def get_sensitive_hosts(self) -> list[tuple[int, int]]:
        return self._sensitive_hosts

    def close(self) -> None:
        self._env.close()
