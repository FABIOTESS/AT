"""Probing-based attack agent (Approach 1).

Systematically scans and exploits the first discovered vulnerability.
Breadth-first enumeration strategy — fast but noisy.
"""

import numpy as np

from cybersim.agents.base import AttackAgent


class ProbingAgent(AttackAgent):
    """Probing agent: cycles through all actions sequentially.

    Emulates breadth-first scanning by trying every available action
    in order, exploiting whatever it finds.
    """

    def __init__(self, num_actions: int) -> None:
        super().__init__(name="probing")
        self._num_actions = num_actions
        self._action_index = 0

    def reset(self) -> None:
        self._action_index = 0

    def act(self, observation: np.ndarray) -> int:
        action = self._action_index % max(self._num_actions, 1)
        self._action_index += 1
        return int(action)
