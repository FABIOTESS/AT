"""Abstract base class for attack agents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from cybersim.simulators.base import SimulatorInterface


@dataclass
class StepRecord:
    """Record of a single step in an episode."""
    step: int
    action: int
    reward: float
    done: bool


@dataclass
class EpisodeResult:
    """Result from running a complete episode."""
    compromised: bool
    steps: int
    total_reward: float
    stealth_score: float | None = None
    timeline: list[StepRecord] = field(default_factory=list)


class AttackAgent(ABC):
    """Abstract base class for cybersecurity attack agents.

    Subclasses implement act() and reset(). The episode loop
    is shared — no more duplicated simulation code.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def act(self, observation: np.ndarray) -> int:
        """Choose an action given the current observation."""

    @abstractmethod
    def reset(self) -> None:
        """Reset agent internal state for a new episode."""

    def run_episode(
        self,
        sim: SimulatorInterface,
        max_steps: int = 500,
    ) -> EpisodeResult:
        """Run a single episode. Shared by all agents."""
        self.reset()
        obs = sim.reset()
        total_reward = 0.0
        timeline: list[StepRecord] = []
        info = {}
        step = 0

        for step in range(1, max_steps + 1):
            action = self.act(obs)
            obs, reward, done, info = sim.step(action)
            total_reward += reward

            timeline.append(StepRecord(
                step=step,
                action=action,
                reward=reward,
                done=done,
            ))

            if done:
                break

        compromised = info.get("goal_reached", total_reward > 0) if info else total_reward > 0
        stealth_score = self.compute_stealth_score(timeline) if hasattr(self, "compute_stealth_score") else None

        return EpisodeResult(
            compromised=compromised,
            steps=step,
            total_reward=total_reward,
            stealth_score=stealth_score,
            timeline=timeline,
        )
