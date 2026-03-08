"""Benchmark plot generation using matplotlib."""

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cybersim.benchmarks.metrics import AgentMetrics

AGENT_COLORS = {
    "ppo": "#2196F3",
    "probing": "#4CAF50",
    "killchain": "#FF9800",
    "stealth": "#9C27B0",
}


def _get_color(agent_name: str) -> str:
    return AGENT_COLORS.get(agent_name, "#607D8B")


def _group_by_config(metrics: list[AgentMetrics]) -> dict[str, list[AgentMetrics]]:
    grouped: dict[str, list[AgentMetrics]] = defaultdict(list)
    for m in metrics:
        grouped[m.config_name].append(m)
    return grouped


def _group_by_agent(metrics: list[AgentMetrics]) -> dict[str, list[AgentMetrics]]:
    grouped: dict[str, list[AgentMetrics]] = defaultdict(list)
    for m in metrics:
        grouped[m.agent_name].append(m)
    return grouped


def plot_compromise_rates(metrics: list[AgentMetrics], output_dir: Path) -> None:
    by_config = _group_by_config(metrics)
    configs = list(by_config.keys())
    agents = sorted({m.agent_name for m in metrics})

    x = np.arange(len(configs))
    width = 0.8 / max(len(agents), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, agent in enumerate(agents):
        rates = []
        for config in configs:
            match = [m for m in by_config[config] if m.agent_name == agent]
            rates.append(match[0].compromise_rate * 100 if match else 0)
        ax.bar(x + i * width, rates, width, label=agent.title(), color=_get_color(agent))

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Compromise Rate (%)")
    ax.set_title("Compromise Rate by Agent and Configuration")
    ax.set_xticks(x + width * (len(agents) - 1) / 2)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(output_dir / "compromise_rates.png", dpi=150)
    plt.close(fig)


def plot_steps_comparison(metrics: list[AgentMetrics], output_dir: Path) -> None:
    by_agent = _group_by_agent(metrics)
    agents = sorted(by_agent.keys())

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(agents))
    avg_steps = [np.mean([m.avg_steps for m in by_agent[a]]) for a in agents]
    std_steps = [np.mean([m.std_steps for m in by_agent[a]]) for a in agents]
    colors = [_get_color(a) for a in agents]

    ax.bar(x, avg_steps, yerr=std_steps, capsize=5, color=colors)
    ax.set_xlabel("Agent")
    ax.set_ylabel("Average Steps to Compromise")
    ax.set_title("Steps Comparison Across Agents")
    ax.set_xticks(x)
    ax.set_xticklabels([a.title() for a in agents])
    fig.tight_layout()
    fig.savefig(output_dir / "steps_comparison.png", dpi=150)
    plt.close(fig)


def plot_time_comparison(metrics: list[AgentMetrics], output_dir: Path) -> None:
    by_agent = _group_by_agent(metrics)
    agents = sorted(by_agent.keys())

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(agents))
    avg_rewards = [np.mean([m.avg_reward for m in by_agent[a]]) for a in agents]
    colors = [_get_color(a) for a in agents]

    ax.bar(x, avg_rewards, color=colors)
    ax.set_xlabel("Agent")
    ax.set_ylabel("Average Reward")
    ax.set_title("Reward Comparison Across Agents")
    ax.set_xticks(x)
    ax.set_xticklabels([a.title() for a in agents])
    fig.tight_layout()
    fig.savefig(output_dir / "time_comparison.png", dpi=150)
    plt.close(fig)


def plot_scaling(metrics: list[AgentMetrics], output_dir: Path) -> None:
    by_agent = _group_by_agent(metrics)
    agents = sorted(by_agent.keys())
    configs = sorted({m.config_name for m in metrics})

    fig, ax = plt.subplots(figsize=(10, 6))
    for agent in agents:
        agent_metrics = by_agent[agent]
        config_steps = {m.config_name: m.avg_steps for m in agent_metrics}
        steps = [config_steps.get(c, 0) for c in configs]
        ax.plot(configs, steps, marker="o", label=agent.title(), color=_get_color(agent))

    ax.set_xlabel("Configuration (increasing complexity)")
    ax.set_ylabel("Average Steps")
    ax.set_title("Scaling: Steps vs Network Complexity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "scaling.png", dpi=150)
    plt.close(fig)


def plot_agent_radar(metrics: list[AgentMetrics], output_dir: Path) -> None:
    by_agent = _group_by_agent(metrics)
    agents = sorted(by_agent.keys())

    categories = ["Compromise\nRate", "Speed\n(inv. steps)", "Reward", "Consistency"]
    num_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for agent in agents:
        agent_data = by_agent[agent]
        cr = np.mean([m.compromise_rate for m in agent_data])
        avg_steps = np.mean([m.avg_steps for m in agent_data])
        speed = 1.0 / max(avg_steps, 1) * 100
        reward = max(np.mean([m.avg_reward for m in agent_data]), 0) / 10
        consistency = 1.0 - min(np.mean([m.std_steps for m in agent_data]) / 50, 1.0)

        values = [cr, min(speed, 1.0), min(reward, 1.0), consistency]
        values += values[:1]

        ax.plot(angles, values, label=agent.title(), color=_get_color(agent))
        ax.fill(angles, values, alpha=0.1, color=_get_color(agent))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.1)
    ax.set_title("Agent Comparison Radar", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(output_dir / "agent_radar.png", dpi=150)
    plt.close(fig)


def plot_stealth_tradeoff(metrics: list[AgentMetrics], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for m in metrics:
        stealth = m.avg_stealth_score if m.avg_stealth_score is not None else 0.0
        ax.scatter(
            stealth,
            m.compromise_rate * 100,
            s=100,
            color=_get_color(m.agent_name),
            label=f"{m.agent_name.title()} ({m.config_name})",
            alpha=0.8,
            edgecolors="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Stealth Score")
    ax.set_ylabel("Compromise Rate (%)")
    ax.set_title("Stealth vs Compromise Trade-off")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "stealth_tradeoff.png", dpi=150)
    plt.close(fig)


def generate_all_plots(metrics: list[AgentMetrics], output_dir: Path) -> None:
    """Generate all benchmark plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_compromise_rates(metrics, output_dir)
    plot_steps_comparison(metrics, output_dir)
    plot_time_comparison(metrics, output_dir)
    plot_scaling(metrics, output_dir)
    plot_agent_radar(metrics, output_dir)
    plot_stealth_tradeoff(metrics, output_dir)
