"""Benchmark metric calculations."""

import statistics
from dataclasses import dataclass

from cybersim.agents.base import EpisodeResult


@dataclass
class AgentMetrics:
    """Aggregated metrics for one agent across multiple episodes."""
    agent_name: str
    config_name: str
    compromise_rate: float
    avg_steps: float
    std_steps: float
    avg_reward: float
    std_reward: float
    avg_stealth_score: float | None
    total_episodes: int


def compute_metrics(
    results: list[EpisodeResult],
    agent_name: str = "",
    config_name: str = "",
) -> AgentMetrics:
    """Compute aggregated metrics from a list of episode results."""
    if not results:
        return AgentMetrics(
            agent_name=agent_name,
            config_name=config_name,
            compromise_rate=0.0,
            avg_steps=0.0,
            std_steps=0.0,
            avg_reward=0.0,
            std_reward=0.0,
            avg_stealth_score=None,
            total_episodes=0,
        )

    n = len(results)
    successes = sum(1 for r in results if r.compromised)
    steps = [r.steps for r in results]
    rewards = [r.total_reward for r in results]
    stealth_scores = [r.stealth_score for r in results if r.stealth_score is not None]

    return AgentMetrics(
        agent_name=agent_name,
        config_name=config_name,
        compromise_rate=successes / n,
        avg_steps=statistics.mean(steps),
        std_steps=statistics.stdev(steps) if n > 1 else 0.0,
        avg_reward=statistics.mean(rewards),
        std_reward=statistics.stdev(rewards) if n > 1 else 0.0,
        avg_stealth_score=statistics.mean(stealth_scores) if stealth_scores else None,
        total_episodes=n,
    )
