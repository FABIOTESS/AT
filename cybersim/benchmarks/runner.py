"""Benchmark runner: executes all agents across all configs."""

import argparse
import time
from pathlib import Path

from cybersim.agents.base import AttackAgent, EpisodeResult
from cybersim.agents.killchain_agent import KillChainAgent
from cybersim.agents.probing_agent import ProbingAgent
from cybersim.agents.stealth_agent import StealthAgent
from cybersim.benchmarks.metrics import AgentMetrics, compute_metrics
from cybersim.simulators.nasim_adapter import NaSimAdapter
from cybersim.utils.logger import get_logger

logger = get_logger("benchmark")


class BenchmarkRunner:
    """Runs all agents across all configs and collects metrics."""

    def __init__(
        self,
        config_paths: list[str],
        num_episodes: int = 100,
        max_steps: int = 500,
        include_ppo: bool = False,
        ppo_timesteps: int = 100_000,
    ) -> None:
        self._config_paths = config_paths
        self._num_episodes = num_episodes
        self._max_steps = max_steps
        self._include_ppo = include_ppo
        self._ppo_timesteps = ppo_timesteps

    def _create_agents(self, num_actions: int, sim: NaSimAdapter) -> list[AttackAgent]:
        agents: list[AttackAgent] = [
            ProbingAgent(num_actions=num_actions),
            KillChainAgent(num_actions=num_actions),
            StealthAgent(num_actions=num_actions),
        ]
        if self._include_ppo:
            from cybersim.agents.ppo_agent import PPOAgent
            ppo = PPOAgent(sim=sim, total_timesteps=self._ppo_timesteps)
            logger.info("Training PPO agent...")
            ppo.train(sim)
            agents.append(ppo)
        return agents

    def run(self) -> list[AgentMetrics]:
        """Run benchmarks and return metrics for all agent/config pairs."""
        all_metrics: list[AgentMetrics] = []

        for config_path in self._config_paths:
            config_name = Path(config_path).stem
            logger.info(f"Benchmarking config: {config_name}")

            sim = NaSimAdapter(config_path)
            num_actions = sim.get_action_space().n
            agents = self._create_agents(num_actions, sim)

            for agent in agents:
                logger.info(f"  Running {agent.name} x{self._num_episodes} episodes")
                episodes: list[EpisodeResult] = []

                for _ in range(self._num_episodes):
                    result = agent.run_episode(sim, max_steps=self._max_steps)
                    episodes.append(result)

                metrics = compute_metrics(episodes, agent.name, config_name)
                all_metrics.append(metrics)
                logger.info(
                    f"    {agent.name}: compromise={metrics.compromise_rate:.0%} "
                    f"avg_steps={metrics.avg_steps:.1f}"
                )

            sim.close()

        return all_metrics


def main() -> None:
    """CLI entry point for running benchmarks."""
    parser = argparse.ArgumentParser(description="Run CyberSim benchmarks")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Config YAML paths (default: all in configs/baselines + configs/generated)",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per agent/config")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--output", type=str, default="assets", help="Output directory for plots")
    parser.add_argument("--include-ppo", action="store_true", help="Include PPO agent (slow)")
    parser.add_argument("--ppo-timesteps", type=int, default=100_000, help="PPO training steps")
    args = parser.parse_args()

    config_paths = args.configs
    if not config_paths:
        config_dirs = [Path("cybersim/configs/baselines"), Path("cybersim/configs/generated")]
        config_paths = []
        for d in config_dirs:
            if d.exists():
                config_paths.extend(str(p) for p in sorted(d.glob("*.yaml")))

    if not config_paths:
        logger.error("No config files found. Add YAMLs to cybersim/configs/baselines/")
        return

    # Import here so plots is only needed when running CLI
    from cybersim.benchmarks.plots import generate_all_plots

    runner = BenchmarkRunner(
        config_paths=config_paths,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        include_ppo=args.include_ppo,
        ppo_timesteps=args.ppo_timesteps,
    )
    metrics = runner.run()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_all_plots(metrics, output_dir)
    logger.info(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
