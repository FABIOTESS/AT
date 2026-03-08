"""Abstract base class for cybersecurity simulators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass
class StepResult:
    """Result from a single environment step."""
    observation: np.ndarray
    reward: float
    done: bool
    info: dict


class SimulatorInterface(ABC):
    """Abstract interface for cybersecurity simulation environments.

    Implementations wrap specific simulators (NASim, CyberBattleSim, etc.)
    behind a uniform API so agents are simulator-agnostic.
    """

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset the environment and return initial observation."""

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Take an action. Returns (observation, reward, done, info)."""

    @abstractmethod
    def get_action_space(self) -> gym.Space:
        """Return the action space."""

    @abstractmethod
    def get_observation_space(self) -> gym.Space:
        """Return the observation space."""

    @abstractmethod
    def get_sensitive_hosts(self) -> list[tuple[int, int]]:
        """Return list of sensitive host (subnet, host) tuples."""

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
