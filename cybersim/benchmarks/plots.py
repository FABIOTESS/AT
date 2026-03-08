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

# Logical ordering: tiny → small → medium
_SCALE_ORDER = {"tiny": 0, "small": 1, "medium": 2}


def _get_color(agent_name: str) -> str:
    return AGENT_COLORS.get(agent_name, "#607D8B")


def _pretty_name(config_name: str) -> str:
    """Convert 'tiny_corp' to 'Tiny Corp'."""
    return config_name.replace("_", " ").title()


def _config_sort_key(name: str) -> tuple[int, str]:
    """Sort configs by scale tier then alphabetically."""
    for prefix, order in _SCALE_ORDER.items():
        if name.startswith(prefix):
            return (order, name)
    return (99, name)


def _sorted_configs(metrics: list[AgentMetrics]) -> list[str]:
    """Get config names sorted by complexity tier."""
    configs = sorted({m.config_name for m in metrics}, key=_config_sort_key)
    return configs


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


def _apply_style(ax: plt.Axes) -> None:
    """Apply consistent clean styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)
    ax.title.set_size(13)
    ax.title.set_weight("bold")


def plot_compromise_rates(metrics: list[AgentMetrics], output_dir: Path) -> None:
    """Grouped bar chart: compromise rate per agent x config."""
    by_config = _group_by_config(metrics)
    configs = _sorted_configs(metrics)
    agents = sorted({m.agent_name for m in metrics})

    x = np.arange(len(configs))
    width = 0.8 / max(len(agents), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, agent in enumerate(agents):
        rates = []
        for config in configs:
            match = [m for m in by_config[config] if m.agent_name == agent]
            rates.append(match[0].compromise_rate * 100 if match else 0)
        ax.bar(
            x + i * width, rates, width,
            label=agent.title(), color=_get_color(agent),
            edgecolor="white", linewidth=0.5,
        )

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Compromise Rate (%)")
    ax.set_title("Compromise Rate by Agent and Configuration")
    ax.set_xticks(x + width * (len(agents) - 1) / 2)
    ax.set_xticklabels([_pretty_name(c) for c in configs], rotation=30, ha="right")
    ax.legend(framealpha=0.9, fontsize=10)
    ax.set_ylim(0, 110)
    _apply_style(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "compromise_rates_v2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_steps_comparison(metrics: list[AgentMetrics], output_dir: Path) -> None:
    """Bar chart with error bars: avg steps per agent."""
    by_agent = _group_by_agent(metrics)
    agents = sorted(by_agent.keys())

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(agents))
    avg_steps = [np.mean([m.avg_steps for m in by_agent[a]]) for a in agents]
    std_steps = [np.mean([m.std_steps for m in by_agent[a]]) for a in agents]
    colors = [_get_color(a) for a in agents]

    bars = ax.bar(x, avg_steps, yerr=std_steps, capsize=6, color=colors, edgecolor="white")
    # Add value labels on bars
    for bar, val in zip(bars, avg_steps):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f"{val:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("Agent")
    ax.set_ylabel("Average Steps to Compromise")
    ax.set_title("Steps Comparison Across Agents")
    ax.set_xticks(x)
    ax.set_xticklabels([a.title() for a in agents], fontsize=11)
    _apply_style(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "steps_comparison_v2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_time_comparison(metrics: list[AgentMetrics], output_dir: Path) -> None:
    """Bar chart: average reward per agent."""
    by_agent = _group_by_agent(metrics)
    agents = sorted(by_agent.keys())

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(agents))
    avg_rewards = [np.mean([m.avg_reward for m in by_agent[a]]) for a in agents]
    colors = [_get_color(a) for a in agents]

    bars = ax.bar(x, avg_rewards, color=colors, edgecolor="white")
    for bar, val in zip(bars, avg_rewards):
        y_pos = bar.get_height() + 2 if val >= 0 else bar.get_height() - 8
        ax.text(
            bar.get_x() + bar.get_width() / 2, y_pos,
            f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.axhline(y=0, color="grey", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Average Reward")
    ax.set_title("Reward Comparison Across Agents")
    ax.set_xticks(x)
    ax.set_xticklabels([a.title() for a in agents], fontsize=11)
    _apply_style(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "time_comparison_v2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scaling(metrics: list[AgentMetrics], output_dir: Path) -> None:
    """Line chart: performance vs config complexity (ordered by scale tier)."""
    by_agent = _group_by_agent(metrics)
    agents = sorted(by_agent.keys())
    configs = _sorted_configs(metrics)
    pretty_configs = [_pretty_name(c) for c in configs]

    fig, ax = plt.subplots(figsize=(12, 6))
    for agent in agents:
        agent_metrics = by_agent[agent]
        config_steps = {m.config_name: m.avg_steps for m in agent_metrics}
        steps = [config_steps.get(c, 0) for c in configs]
        ax.plot(
            pretty_configs, steps,
            marker="o", label=agent.title(), color=_get_color(agent),
            linewidth=2, markersize=7,
        )

    ax.set_xlabel("Configuration (increasing complexity)")
    ax.set_ylabel("Average Steps")
    ax.set_title("Scaling: Steps vs Network Complexity")
    ax.legend(framealpha=0.9, fontsize=10)
    plt.xticks(rotation=30, ha="right")
    _apply_style(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "scaling_v2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_agent_radar(metrics: list[AgentMetrics], output_dir: Path) -> None:
    """Radar chart: multi-metric comparison across agents."""
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
        consistency = 1.0 - min(
            np.mean([m.std_steps for m in agent_data]) / 50, 1.0
        )

        values = [cr, min(speed, 1.0), min(reward, 1.0), consistency]
        values += values[:1]

        ax.plot(
            angles, values,
            label=agent.title(), color=_get_color(agent), linewidth=2,
        )
        ax.fill(angles, values, alpha=0.15, color=_get_color(agent))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("Agent Comparison Radar", y=1.08, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "agent_radar_v2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stealth_tradeoff(metrics: list[AgentMetrics], output_dir: Path) -> None:
    """Scatter: stealth score vs compromise rate.

    Only the stealth agent has meaningful stealth scores; other agents are
    shown as aggregate reference markers at x=0 so the chart stays clean.
    """
    by_agent = _group_by_agent(metrics)
    fig, ax = plt.subplots(figsize=(10, 7))

    for agent_name in sorted(by_agent.keys()):
        agent_metrics = by_agent[agent_name]
        has_stealth = any(m.avg_stealth_score is not None for m in agent_metrics)
        color = _get_color(agent_name)

        if has_stealth:
            # Plot each config individually — these have interesting spread
            # Collect points first to compute label offsets that avoid overlap
            points = []
            for m in sorted(agent_metrics, key=lambda x: x.config_name):
                stealth = m.avg_stealth_score if m.avg_stealth_score is not None else 0.0
                points.append((stealth, m.compromise_rate * 100, _pretty_name(m.config_name)))

            first = True
            for i, (sx, sy, name) in enumerate(points):
                ax.scatter(
                    sx, sy, s=140, color=color, alpha=0.85,
                    edgecolors="black", linewidth=0.5, zorder=3,
                    label=agent_name.title() if first else None,
                )
                # Stagger vertical offset for points near each other
                nearby = sum(
                    1 for j, (ox, oy, _) in enumerate(points[:i])
                    if abs(oy - sy) < 8 and abs(ox - sx) < 0.06
                )
                y_off = -6 - nearby * 16  # push labels down for each collision
                ax.annotate(
                    name, (sx, sy),
                    textcoords="offset points", xytext=(8, y_off),
                    fontsize=8, alpha=0.8,
                )
                first = False
        else:
            # Aggregate: single marker at x=0 for non-stealth agents
            avg_cr = np.mean([m.compromise_rate * 100 for m in agent_metrics])
            ax.scatter(
                0, avg_cr, s=180, color=color, alpha=0.85,
                edgecolors="black", linewidth=0.5, zorder=3,
                marker="D", label=f"{agent_name.title()} (avg)",
            )
            ax.annotate(
                f"{avg_cr:.0f}%", (0, avg_cr),
                textcoords="offset points", xytext=(10, 0),
                fontsize=9, fontweight="bold", alpha=0.8,
            )

    ax.set_xlabel("Stealth Score")
    ax.set_ylabel("Compromise Rate (%)")
    ax.set_title("Stealth vs Compromise Trade-off")
    ax.set_xlim(-0.03, None)
    ax.set_ylim(-5, 115)
    ax.legend(fontsize=10, framealpha=0.9, loc="lower right")
    _apply_style(ax)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Vertical separator between stealth=0 cluster and actual scores
    ax.axvline(x=0.02, color="grey", linewidth=0.6, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_dir / "stealth_tradeoff_v2.png", dpi=150, bbox_inches="tight")
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
