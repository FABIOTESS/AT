import pytest
import numpy as np
from cybersim.agents.base import AttackAgent, EpisodeResult
from cybersim.simulators.nasim_adapter import NaSimAdapter


def test_attack_agent_is_abstract():
    with pytest.raises(TypeError):
        AttackAgent(name="test")


def test_episode_result_dataclass():
    result = EpisodeResult(
        compromised=True,
        steps=10,
        total_reward=5.0,
        stealth_score=None,
        timeline=[],
    )
    assert result.compromised is True
    assert result.steps == 10


class DummyAgent(AttackAgent):
    """Minimal agent that always takes action 0."""

    def act(self, observation: np.ndarray) -> int:
        return 0

    def reset(self) -> None:
        pass


def test_run_episode_returns_episode_result():
    sim = NaSimAdapter("cybersim/configs/baselines/tiny.yaml")
    agent = DummyAgent(name="dummy")
    result = agent.run_episode(sim, max_steps=10)
    assert isinstance(result, EpisodeResult)
    assert isinstance(result.compromised, bool)
    assert result.steps <= 10
    assert isinstance(result.total_reward, float)
    sim.close()
