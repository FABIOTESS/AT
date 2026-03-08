"""Cyber Kill Chain attack agent (Approach 2).

Simulates multi-stage attacks following Lockheed Martin's kill chain:
reconnaissance -> weaponization -> delivery -> exploitation -> C2.
"""

import numpy as np

from cybersim.agents.base import AttackAgent

_PHASES = ["recon", "weaponize", "deliver", "exploit", "command_control"]


class KillChainAgent(AttackAgent):
    """Kill chain agent: partitions actions into phases.

    Divides the action space into 5 kill chain phases and progresses
    through them sequentially, advancing to the next phase after
    exhausting actions in the current one.
    """

    def __init__(self, num_actions: int) -> None:
        super().__init__(name="killchain")
        self._num_actions = num_actions
        self._phase_size = max(1, num_actions // len(_PHASES))
        self._current_phase = 0
        self._phase_step = 0

    def reset(self) -> None:
        self._current_phase = 0
        self._phase_step = 0

    def act(self, observation: np.ndarray) -> int:
        phase_start = self._current_phase * self._phase_size
        action = (phase_start + self._phase_step) % self._num_actions

        self._phase_step += 1
        if self._phase_step >= self._phase_size:
            self._phase_step = 0
            self._current_phase = (self._current_phase + 1) % len(_PHASES)

        return int(action)

    @property
    def current_phase(self) -> str:
        return _PHASES[self._current_phase]
