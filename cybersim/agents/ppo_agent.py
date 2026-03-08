"""PPO-based reinforcement learning agent (Approach 0).

Uses Stable-Baselines3 PPO with MLP policy as the RL baseline.
Unlike the original code, this properly uses the NASim environment
instead of CartPole.
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from cybersim.agents.base import AttackAgent
from cybersim.simulators.base import SimulatorInterface
from cybersim.simulators.nasim_adapter import NaSimAdapter


class _IntActionWrapper(gym.ActionWrapper):
    """Convert numpy-int actions to plain Python ints for NASim compatibility."""

    def action(self, action):
        return int(action)


class PPOAgent(AttackAgent):
    """PPO agent wrapping Stable-Baselines3.

    Must be trained before running episodes. Uses the actual
    NASim environment (not CartPole like the original code).
    """

    def __init__(
        self,
        sim: SimulatorInterface,
        total_timesteps: int = 100_000,
        learning_rate: float = 3e-4,
        clip_range: float = 0.2,
        gamma: float = 0.99,
    ) -> None:
        super().__init__(name="ppo")
        self._total_timesteps = total_timesteps
        self._learning_rate = learning_rate
        self._clip_range = clip_range
        self._gamma = gamma
        self._model: PPO | None = None
        self._trained = False

    def train(self, sim: SimulatorInterface) -> None:
        """Train the PPO model on the given simulator."""
        if not isinstance(sim, NaSimAdapter):
            raise TypeError("PPOAgent requires a NaSimAdapter for training")

        import nasim
        env = nasim.load(
            sim._config_path,
            fully_obs=True,
            flat_actions=True,
            flat_obs=True,
        )
        env = _IntActionWrapper(env)
        env = Monitor(env)

        self._model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self._learning_rate,
            clip_range=self._clip_range,
            gamma=self._gamma,
            verbose=0,
        )
        self._model.learn(total_timesteps=self._total_timesteps)
        env.close()
        self._trained = True

    def reset(self) -> None:
        pass

    def act(self, observation: np.ndarray) -> int:
        if self._model is None:
            raise RuntimeError("PPOAgent must be trained before acting. Call train() first.")
        action, _ = self._model.predict(observation, deterministic=True)
        return int(action)
