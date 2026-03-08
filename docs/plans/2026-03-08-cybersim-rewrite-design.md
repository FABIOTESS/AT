# CyberSim Rewrite Design

**Date:** 2026-03-08
**Goal:** Clean rewrite of the AT repository as a modern portfolio piece showcasing the paper "Proving the Utility of Large Language Models in Cybersecurity Simulations"

## Decisions

- **Approach:** Clean rewrite (not incremental refactor)
- **Goal:** Portfolio/showcase piece
- **LLM generation:** Subagent-generated configs shipped with repo (no API integration)
- **Simulator:** Abstract interface with NASim adapter (swappable)
- **Benchmarks:** Reproduce paper tables + new benchmarks, all charts in README
- **Python:** 3.11+, pyproject.toml, type hints, ruff, pytest

## Package Structure

```
cybersim/
├── __init__.py
├── simulators/
│   ├── __init__.py
│   ├── base.py              # Abstract SimulatorInterface
│   └── nasim_adapter.py     # NASim implementation
├── agents/
│   ├── __init__.py
│   ├── base.py              # Abstract AttackAgent + shared episode loop
│   ├── ppo_agent.py         # Approach 0: PPO via Stable-Baselines3
│   ├── probing_agent.py     # Approach 1: Breadth-first scan + exploit
│   ├── killchain_agent.py   # Approach 2: Multi-stage cyber kill chain
│   └── stealth_agent.py     # Approach 3: Privilege escalation + stealth score
├── validation/
│   ├── __init__.py
│   └── topology.py          # YAML topology validation
├── benchmarks/
│   ├── __init__.py
│   ├── runner.py            # BenchmarkRunner
│   ├── metrics.py           # Metric calculations
│   └── plots.py             # Chart generation → assets/
├── configs/
│   ├── generated/           # Subagent-generated YAMLs (9 configs)
│   └── baselines/           # Hand-crafted reference configs
└── utils/
    ├── __init__.py
    ├── logger.py
    └── helpers.py

tests/
├── test_agents.py
├── test_simulators.py
├── test_validation.py
└── test_benchmarks.py

assets/                      # Auto-generated benchmark plots
pyproject.toml
README.md
```

## Core Abstractions

### SimulatorInterface

```python
class SimulatorInterface(ABC):
    @abstractmethod
    def reset(self) -> Observation: ...
    @abstractmethod
    def step(self, action: int) -> tuple[Observation, float, bool, dict]: ...
    @abstractmethod
    def get_action_space(self) -> gym.Space: ...
    @abstractmethod
    def get_observation_space(self) -> gym.Space: ...
    @abstractmethod
    def get_sensitive_hosts(self) -> list[tuple[int, int]]: ...
```

NaSimAdapter wraps NASim behind this interface. Swapping simulators = new adapter class.

### AttackAgent

```python
class AttackAgent(ABC):
    name: str

    @abstractmethod
    def act(self, observation: Observation) -> int: ...
    @abstractmethod
    def reset(self) -> None: ...

    def run_episode(self, sim: SimulatorInterface, max_steps: int = 500) -> EpisodeResult:
        # Shared episode loop — eliminates all duplication
        ...
```

EpisodeResult dataclass: compromised, steps, total_reward, stealth_score, timeline.

### Agents

| Agent | Strategy | Key Metric |
|-------|----------|------------|
| PPO (Approach 0) | Stable-Baselines3 PPO, MLP policy | Baseline RL |
| Probing (Approach 1) | Breadth-first scan, exploit first open door | Speed |
| Kill Chain (Approach 2) | Multi-stage: recon → weaponize → deliver → exploit → C2 | Realism |
| Stealth (Approach 3) | Lateral movement, minimal footprint, partial observability | Stealth score |

## Benchmarking

### Runner

- All agents x all configs, N=100 episodes each
- Single command: `python -m cybersim.benchmarks.runner --output assets/`

### Metrics

- Compromise rate (%)
- Avg steps to compromise
- Avg wall-clock time (s)
- Stealth score (Agent 3 only)
- Std deviation for all

### Plots (saved to assets/, embedded in README)

| File | Type | Shows |
|------|------|-------|
| compromise_rates.png | Grouped bar | Compromise rate per agent x config |
| steps_comparison.png | Box plot | Step distribution per agent |
| time_comparison.png | Bar chart | Wall-clock time per agent |
| scaling.png | Line chart | Performance vs network size |
| agent_radar.png | Radar chart | Multi-metric comparison |
| stealth_tradeoff.png | Scatter | Stealth score vs compromise rate |

## Config Generation

9 YAML configs generated via Claude Code subagents:

| Scale | Subnets | Hosts | Count |
|-------|---------|-------|-------|
| Tiny | 1-2 | 3-5 | 3 |
| Small | 3-4 | 8-12 | 3 |
| Medium | 5-8 | 15-25 | 3 |

Each config has a comment header documenting LLM generation.

## Topology Validation (Paper Figure 2)

- Adjacency matrix symmetry and subnet consistency
- Reachability: every host reachable from attacker
- Vulnerability references valid services
- Sensitive hosts exist in topology
- Firewall rules consistent with adjacency

Returns ValidationResult with pass/fail + issue list.

## Modernisation

- Python 3.11+ (match expressions, modern type syntax)
- gymnasium replaces deprecated gym
- pyproject.toml with pinned dependency ranges
- ruff for linting
- pytest with focused test suite
- Type hints everywhere
- No debug prints
