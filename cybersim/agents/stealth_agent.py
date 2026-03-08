"""Stealth privilege escalation agent (Approach 3).

Prioritises lateral movement with minimal observable footprint.
Implements stealth scoring based on action diversity and penalty patterns.
"""

import numpy as np

from cybersim.agents.base import AttackAgent, StepRecord


class StealthAgent(AttackAgent):
    """Stealth agent: avoids repeated actions, minimises footprint.

    Uses a weighted random strategy biased toward unexplored actions.
    Computes a stealth score based on:
    - Action diversity (fewer repeated actions = stealthier)
    - Negative reward avoidance (penalties suggest detection)
    """

    def __init__(self, num_actions: int, exploration_decay: float = 0.95) -> None:
        super().__init__(name="stealth")
        self._num_actions = num_actions
        self._exploration_decay = exploration_decay
        self._action_counts: np.ndarray = np.zeros(num_actions)
        self._rng = np.random.default_rng(42)

    def reset(self) -> None:
        self._action_counts = np.zeros(self._num_actions)

    def act(self, observation: np.ndarray) -> int:
        weights = 1.0 / (self._action_counts + 1.0)
        weights = weights ** self._exploration_decay
        probabilities = weights / weights.sum()

        action = int(self._rng.choice(self._num_actions, p=probabilities))
        self._action_counts[action] += 1
        return action

    def compute_stealth_score(self, timeline: list[StepRecord]) -> float:
        """Compute stealth score in [0, 1]. Higher = stealthier.

        Components:
        - diversity: ratio of unique actions to total steps
        - penalty_avoidance: fraction of steps without negative reward
        """
        if not timeline:
            return 1.0

        total_steps = len(timeline)
        unique_actions = len({s.action for s in timeline})
        diversity = min(unique_actions / max(total_steps, 1), 1.0)

        penalties = sum(1 for s in timeline if s.reward < 0)
        penalty_avoidance = 1.0 - (penalties / total_steps)

        return round(0.4 * diversity + 0.6 * penalty_avoidance, 4)
