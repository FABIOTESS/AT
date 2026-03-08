# CyberSim Rewrite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clean rewrite of the AT repo as a modern Python portfolio piece showcasing LLM-driven cybersecurity simulations.

**Architecture:** Abstract simulator interface (NASim adapter), base AttackAgent with shared episode loop, 4 concrete agents, topology validator, automated benchmark runner with 6 plot types. All generated configs ship with the repo.

**Tech Stack:** Python 3.11+, gymnasium, nasim, stable-baselines3, numpy, matplotlib, pyyaml, pytest, ruff

---

### Task 1: Project Scaffold & pyproject.toml

**Files:**
- Create: `cybersim/__init__.py`
- Create: `cybersim/simulators/__init__.py`
- Create: `cybersim/agents/__init__.py`
- Create: `cybersim/validation/__init__.py`
- Create: `cybersim/benchmarks/__init__.py`
- Create: `cybersim/configs/generated/.gitkeep`
- Create: `cybersim/configs/baselines/.gitkeep`
- Create: `cybersim/utils/__init__.py`
- Create: `tests/__init__.py`
- Create: `assets/.gitkeep`
- Create: `pyproject.toml`

**Step 1: Create directory structure**

```bash
mkdir -p cybersim/{simulators,agents,validation,benchmarks,configs/{generated,baselines},utils}
mkdir -p tests assets
touch cybersim/__init__.py cybersim/simulators/__init__.py cybersim/agents/__init__.py
touch cybersim/validation/__init__.py cybersim/benchmarks/__init__.py cybersim/utils/__init__.py
touch tests/__init__.py
touch cybersim/configs/generated/.gitkeep cybersim/configs/baselines/.gitkeep assets/.gitkeep
```

**Step 2: Write pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "cybersim"
version = "1.0.0"
description = "LLM-driven cybersecurity simulation framework"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [
    {name = "Fabio Rovai", email = "fabio@thetesseractacademy.com"},
    {name = "Stylianos Kampakis", email = "stelios@thetesseractacademy.com"},
]

dependencies = [
    "nasim>=0.9",
    "stable-baselines3>=2.0",
    "gymnasium>=0.29",
    "numpy>=1.24",
    "matplotlib>=3.7",
    "pyyaml>=6.0",
    "pandas>=2.0",
    "tqdm>=4.65",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.1",
]

[project.scripts]
cybersim-benchmark = "cybersim.benchmarks.runner:main"

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 3: Write cybersim/__init__.py with version**

```python
"""CyberSim: LLM-driven cybersecurity simulation framework."""

__version__ = "1.0.0"
```

**Step 4: Commit**

```bash
git add cybersim/ tests/ assets/ pyproject.toml
git commit -m "feat: scaffold cybersim package with pyproject.toml"
```

---

### Task 2: Simulator Interface & NASim Adapter

**Files:**
- Create: `cybersim/simulators/base.py`
- Create: `cybersim/simulators/nasim_adapter.py`
- Test: `tests/test_simulators.py`

**Step 1: Write the failing test**

```python
# tests/test_simulators.py
import pytest
from cybersim.simulators.base import SimulatorInterface
from cybersim.simulators.nasim_adapter import NaSimAdapter


def test_simulator_interface_is_abstract():
    with pytest.raises(TypeError):
        SimulatorInterface()


def test_nasim_adapter_loads_tiny_config():
    adapter = NaSimAdapter("cybersim/configs/baselines/tiny.yaml")
    obs = adapter.reset()
    assert obs is not None
    assert len(obs.shape) >= 1


def test_nasim_adapter_step():
    adapter = NaSimAdapter("cybersim/configs/baselines/tiny.yaml")
    adapter.reset()
    obs, reward, done, info = adapter.step(0)
    assert obs is not None
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_nasim_adapter_spaces():
    adapter = NaSimAdapter("cybersim/configs/baselines/tiny.yaml")
    action_space = adapter.get_action_space()
    obs_space = adapter.get_observation_space()
    assert action_space is not None
    assert obs_space is not None


def test_nasim_adapter_sensitive_hosts():
    adapter = NaSimAdapter("cybersim/configs/baselines/tiny.yaml")
    hosts = adapter.get_sensitive_hosts()
    assert isinstance(hosts, list)
    assert len(hosts) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_simulators.py -v`
Expected: FAIL (modules don't exist yet)

**Step 3: Copy tiny.yaml to baselines**

Copy the existing `config/tiny.yaml` to `cybersim/configs/baselines/tiny.yaml` as the reference config.

**Step 4: Write SimulatorInterface**

```python
# cybersim/simulators/base.py
"""Abstract base class for cybersecurity simulators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass
class StepResult:
    """Result from a single environment step."""
    observation: np.ndarray
    reward: float
    done: bool
    info: dict


class SimulatorInterface(ABC):
    """Abstract interface for cybersecurity simulation environments.

    Implementations wrap specific simulators (NASim, CyberBattleSim, etc.)
    behind a uniform API so agents are simulator-agnostic.
    """

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset the environment and return initial observation."""

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Take an action. Returns (observation, reward, done, info)."""

    @abstractmethod
    def get_action_space(self) -> gym.Space:
        """Return the action space."""

    @abstractmethod
    def get_observation_space(self) -> gym.Space:
        """Return the observation space."""

    @abstractmethod
    def get_sensitive_hosts(self) -> list[tuple[int, int]]:
        """Return list of sensitive host (subnet, host) tuples."""

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
```

**Step 5: Write NaSimAdapter**

```python
# cybersim/simulators/nasim_adapter.py
"""NASim adapter implementing SimulatorInterface."""

import gymnasium as gym
import nasim
import numpy as np
import yaml

from cybersim.simulators.base import SimulatorInterface


class NaSimAdapter(SimulatorInterface):
    """Wraps NASim behind the SimulatorInterface."""

    def __init__(self, config_path: str) -> None:
        self._config_path = config_path
        self._sensitive_hosts = self._parse_sensitive_hosts()
        self._env = nasim.load(
            config_path,
            fully_obs=True,
            flat_actions=True,
            flat_obs=True,
        )

    def _parse_sensitive_hosts(self) -> list[tuple[int, int]]:
        with open(self._config_path) as f:
            config = yaml.safe_load(f)
        raw = config.get("sensitive_hosts", {})
        hosts = []
        for key in raw:
            # Keys are strings like "(2, 0)" in YAML
            if isinstance(key, str):
                cleaned = key.strip("() ")
                parts = [int(x.strip()) for x in cleaned.split(",")]
                hosts.append(tuple(parts))
            elif isinstance(key, tuple):
                hosts.append(key)
        return hosts

    def reset(self) -> np.ndarray:
        result = self._env.reset()
        if isinstance(result, tuple):
            return result[0]
        return result

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        result = self._env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, float(reward), terminated or truncated, info
        obs, reward, done, info = result
        return obs, float(reward), done, info

    def get_action_space(self) -> gym.Space:
        return self._env.action_space

    def get_observation_space(self) -> gym.Space:
        return self._env.observation_space

    def get_sensitive_hosts(self) -> list[tuple[int, int]]:
        return self._sensitive_hosts

    def close(self) -> None:
        self._env.close()
```

**Step 6: Run tests**

Run: `pytest tests/test_simulators.py -v`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add cybersim/simulators/ cybersim/configs/baselines/tiny.yaml tests/test_simulators.py
git commit -m "feat: add SimulatorInterface and NaSimAdapter"
```

---

### Task 3: Agent Base Class & Data Models

**Files:**
- Create: `cybersim/agents/base.py`
- Test: `tests/test_agents.py`

**Step 1: Write the failing test**

```python
# tests/test_agents.py
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents.py -v`
Expected: FAIL (module doesn't exist)

**Step 3: Write AttackAgent base class**

```python
# cybersim/agents/base.py
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
```

**Step 4: Run tests**

Run: `pytest tests/test_agents.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add cybersim/agents/base.py tests/test_agents.py
git commit -m "feat: add AttackAgent base class with shared episode loop"
```

---

### Task 4: Probing Agent (Approach 1)

**Files:**
- Create: `cybersim/agents/probing_agent.py`
- Modify: `tests/test_agents.py`

**Step 1: Add failing test to tests/test_agents.py**

```python
from cybersim.agents.probing_agent import ProbingAgent


def test_probing_agent_runs_episode():
    sim = NaSimAdapter("cybersim/configs/baselines/tiny.yaml")
    agent = ProbingAgent()
    result = agent.run_episode(sim, max_steps=100)
    assert isinstance(result, EpisodeResult)
    assert result.steps >= 1
    sim.close()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents.py::test_probing_agent_runs_episode -v`
Expected: FAIL (module doesn't exist)

**Step 3: Write ProbingAgent**

```python
# cybersim/agents/probing_agent.py
"""Probing-based attack agent (Approach 1).

Systematically scans and exploits the first discovered vulnerability.
Breadth-first enumeration strategy — fast but noisy.
"""

import numpy as np

from cybersim.agents.base import AttackAgent


class ProbingAgent(AttackAgent):
    """Probing agent: cycles through actions sequentially.

    Emulates breadth-first scanning by trying each action in order,
    exploiting the first open vulnerability found.
    """

    def __init__(self) -> None:
        super().__init__(name="probing")
        self._action_index = 0
        self._num_actions = 0

    def reset(self) -> None:
        self._action_index = 0

    def act(self, observation: np.ndarray) -> int:
        if self._num_actions == 0:
            # Infer action space size from first call — set lazily
            self._num_actions = max(1, observation.shape[0])
        action = self._action_index % self._num_actions
        self._action_index += 1
        return action
```

Note: The agent needs the action space size. We update the base class `run_episode` to pass it, OR we set it lazily. The above uses lazy inference from observation shape, but a better approach is to accept the simulator's action space. Let's refine:

```python
# cybersim/agents/probing_agent.py
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

    def __init__(self, num_actions: int = 0) -> None:
        super().__init__(name="probing")
        self._num_actions = num_actions
        self._action_index = 0

    def reset(self) -> None:
        self._action_index = 0

    def act(self, observation: np.ndarray) -> int:
        action = self._action_index % max(self._num_actions, 1)
        self._action_index += 1
        return action
```

Update the test to pass num_actions from the simulator:

```python
def test_probing_agent_runs_episode():
    sim = NaSimAdapter("cybersim/configs/baselines/tiny.yaml")
    num_actions = sim.get_action_space().n
    agent = ProbingAgent(num_actions=num_actions)
    result = agent.run_episode(sim, max_steps=100)
    assert isinstance(result, EpisodeResult)
    assert result.steps >= 1
    sim.close()
```

**Step 4: Run test**

Run: `pytest tests/test_agents.py::test_probing_agent_runs_episode -v`
Expected: PASS

**Step 5: Commit**

```bash
git add cybersim/agents/probing_agent.py tests/test_agents.py
git commit -m "feat: add ProbingAgent (approach 1)"
```

---

### Task 5: Kill Chain Agent (Approach 2)

**Files:**
- Create: `cybersim/agents/killchain_agent.py`
- Modify: `tests/test_agents.py`

**Step 1: Add failing test**

```python
from cybersim.agents.killchain_agent import KillChainAgent


def test_killchain_agent_runs_episode():
    sim = NaSimAdapter("cybersim/configs/baselines/tiny.yaml")
    num_actions = sim.get_action_space().n
    agent = KillChainAgent(num_actions=num_actions)
    result = agent.run_episode(sim, max_steps=100)
    assert isinstance(result, EpisodeResult)
    assert result.steps >= 1
    sim.close()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents.py::test_killchain_agent_runs_episode -v`
Expected: FAIL

**Step 3: Write KillChainAgent**

```python
# cybersim/agents/killchain_agent.py
"""Cyber Kill Chain attack agent (Approach 2).

Simulates multi-stage attacks following Lockheed Martin's kill chain:
reconnaissance → weaponization → delivery → exploitation → C2.
"""

import numpy as np

from cybersim.agents.base import AttackAgent

# Kill chain phases and what fraction of the action space each covers
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

        return action

    @property
    def current_phase(self) -> str:
        return _PHASES[self._current_phase]
```

**Step 4: Run test**

Run: `pytest tests/test_agents.py::test_killchain_agent_runs_episode -v`
Expected: PASS

**Step 5: Commit**

```bash
git add cybersim/agents/killchain_agent.py tests/test_agents.py
git commit -m "feat: add KillChainAgent (approach 2)"
```

---

### Task 6: Stealth Agent (Approach 3) with Stealth Scoring

**Files:**
- Create: `cybersim/agents/stealth_agent.py`
- Modify: `tests/test_agents.py`

**Step 1: Add failing test**

```python
from cybersim.agents.stealth_agent import StealthAgent


def test_stealth_agent_runs_episode():
    sim = NaSimAdapter("cybersim/configs/baselines/tiny.yaml")
    num_actions = sim.get_action_space().n
    agent = StealthAgent(num_actions=num_actions)
    result = agent.run_episode(sim, max_steps=100)
    assert isinstance(result, EpisodeResult)
    assert result.stealth_score is not None
    assert 0.0 <= result.stealth_score <= 1.0
    sim.close()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents.py::test_stealth_agent_runs_episode -v`
Expected: FAIL

**Step 3: Write StealthAgent**

```python
# cybersim/agents/stealth_agent.py
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
        # Inverse frequency weighting — prefer unexplored actions
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

        # Weighted combination
        return round(0.4 * diversity + 0.6 * penalty_avoidance, 4)
```

**Step 4: Run test**

Run: `pytest tests/test_agents.py::test_stealth_agent_runs_episode -v`
Expected: PASS

**Step 5: Commit**

```bash
git add cybersim/agents/stealth_agent.py tests/test_agents.py
git commit -m "feat: add StealthAgent with stealth scoring (approach 3)"
```

---

### Task 7: PPO Agent (Approach 0)

**Files:**
- Create: `cybersim/agents/ppo_agent.py`
- Modify: `tests/test_agents.py`

**Step 1: Add failing test**

```python
from cybersim.agents.ppo_agent import PPOAgent


def test_ppo_agent_trains_and_runs():
    sim = NaSimAdapter("cybersim/configs/baselines/tiny.yaml")
    agent = PPOAgent(sim=sim, total_timesteps=500)
    agent.train(sim)
    result = agent.run_episode(sim, max_steps=100)
    assert isinstance(result, EpisodeResult)
    sim.close()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agents.py::test_ppo_agent_trains_and_runs -v`
Expected: FAIL

**Step 3: Write PPOAgent**

```python
# cybersim/agents/ppo_agent.py
"""PPO-based reinforcement learning agent (Approach 0).

Uses Stable-Baselines3 PPO with MLP policy as the RL baseline.
Unlike the original code, this properly uses the NASim environment
instead of CartPole.
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from cybersim.agents.base import AttackAgent
from cybersim.simulators.base import SimulatorInterface
from cybersim.simulators.nasim_adapter import NaSimAdapter


class PPOAgent(AttackAgent):
    """PPO agent wrapping Stable-Baselines3.

    Must be trained before running episodes. Uses the actual
    NASim environment (not CartPole like the original code).
    """

    def __init__(
        self,
        sim: SimulatorInterface,
        total_timesteps: int = 100_000,
        learning_rate: float = 3e-4,
        clip_range: float = 0.2,
        gamma: float = 0.99,
    ) -> None:
        super().__init__(name="ppo")
        self._total_timesteps = total_timesteps
        self._learning_rate = learning_rate
        self._clip_range = clip_range
        self._gamma = gamma
        self._model: PPO | None = None
        self._trained = False

    def train(self, sim: SimulatorInterface) -> None:
        """Train the PPO model on the given simulator."""
        if not isinstance(sim, NaSimAdapter):
            raise TypeError("PPOAgent requires a NaSimAdapter for training")

        # Create a fresh env for training (SB3 manages its own env)
        import nasim
        env = nasim.load(
            sim._config_path,
            fully_obs=True,
            flat_actions=True,
            flat_obs=True,
        )
        env = Monitor(env)

        self._model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self._learning_rate,
            clip_range=self._clip_range,
            gamma=self._gamma,
            verbose=0,
        )
        self._model.learn(total_timesteps=self._total_timesteps)
        env.close()
        self._trained = True

    def reset(self) -> None:
        pass  # PPO state is in the model, not per-episode

    def act(self, observation: np.ndarray) -> int:
        if self._model is None:
            raise RuntimeError("PPOAgent must be trained before acting. Call train() first.")
        action, _ = self._model.predict(observation, deterministic=True)
        return int(action)
```

**Step 4: Run test**

Run: `pytest tests/test_agents.py::test_ppo_agent_trains_and_runs -v`
Expected: PASS (may take a few seconds for training)

**Step 5: Commit**

```bash
git add cybersim/agents/ppo_agent.py tests/test_agents.py
git commit -m "feat: add PPOAgent using actual NASim env (approach 0)"
```

---

### Task 8: Topology Validator

**Files:**
- Create: `cybersim/validation/topology.py`
- Test: `tests/test_validation.py`

**Step 1: Write the failing test**

```python
# tests/test_validation.py
import pytest
from cybersim.validation.topology import validate_config, ValidationResult


def test_valid_config_passes():
    result = validate_config("cybersim/configs/baselines/tiny.yaml")
    assert isinstance(result, ValidationResult)
    assert result.valid is True
    assert len(result.issues) == 0


def test_invalid_topology_detected(tmp_path):
    # Asymmetric adjacency matrix
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("""
subnets: [1, 1]
topology: [[ 1, 1, 0],
           [ 1, 1, 1],
           [ 1, 0, 1]]
sensitive_hosts:
  (1, 0): 100
os: [linux]
services: [ssh]
processes: [tomcat]
exploits:
  e_ssh:
    service: ssh
    os: linux
    prob: 0.8
    cost: 1
    access: user
privilege_escalation:
  pe_tomcat:
    process: tomcat
    os: linux
    prob: 1.0
    cost: 1
    access: root
service_scan_cost: 1
os_scan_cost: 1
subnet_scan_cost: 1
process_scan_cost: 1
host_configurations:
  (1, 0):
    os: linux
    services: [ssh]
    processes: [tomcat]
firewall: {}
step_limit: 1000
""")
    result = validate_config(str(bad_config))
    assert result.valid is False
    assert any("symmetric" in issue.lower() for issue in result.issues)


def test_missing_sensitive_host_detected(tmp_path):
    bad_config = tmp_path / "bad2.yaml"
    bad_config.write_text("""
subnets: [1]
topology: [[ 1, 1],
           [ 1, 1]]
sensitive_hosts:
  (99, 0): 100
os: [linux]
services: [ssh]
processes: [tomcat]
exploits:
  e_ssh:
    service: ssh
    os: linux
    prob: 0.8
    cost: 1
    access: user
privilege_escalation:
  pe_tomcat:
    process: tomcat
    os: linux
    prob: 1.0
    cost: 1
    access: root
service_scan_cost: 1
os_scan_cost: 1
subnet_scan_cost: 1
process_scan_cost: 1
host_configurations: {}
firewall: {}
step_limit: 1000
""")
    result = validate_config(str(bad_config))
    assert result.valid is False
    assert any("sensitive" in issue.lower() for issue in result.issues)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_validation.py -v`
Expected: FAIL

**Step 3: Write topology validator**

```python
# cybersim/validation/topology.py
"""YAML topology validation implementing the paper's Figure 2 checks."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ValidationResult:
    """Result of validating a YAML config."""
    valid: bool
    issues: list[str] = field(default_factory=list)


def validate_config(config_path: str) -> ValidationResult:
    """Validate a NASim YAML configuration file.

    Checks:
    - Adjacency matrix is symmetric
    - Subnet count matches topology dimensions
    - Sensitive hosts reference valid (subnet, host) pairs
    - Exploits reference defined services
    - Privilege escalation references defined processes
    """
    issues: list[str] = []
    path = Path(config_path)

    if not path.exists():
        return ValidationResult(valid=False, issues=[f"File not found: {config_path}"])

    with open(path) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        return ValidationResult(valid=False, issues=["Config is not a valid YAML dictionary"])

    subnets = config.get("subnets", [])
    topology = config.get("topology", [])
    sensitive_hosts = config.get("sensitive_hosts", {})
    services = config.get("services", [])
    processes = config.get("processes", [])
    exploits = config.get("exploits", {})
    priv_esc = config.get("privilege_escalation", {})
    host_configs = config.get("host_configurations", {})

    # Check topology dimensions
    num_subnets = len(subnets)
    expected_dim = num_subnets + 1  # includes internet node
    if len(topology) != expected_dim:
        issues.append(
            f"Topology has {len(topology)} rows, expected {expected_dim} "
            f"({num_subnets} subnets + 1 internet node)"
        )

    # Check topology symmetry
    for i, row in enumerate(topology):
        if len(row) != len(topology):
            issues.append(f"Topology row {i} has {len(row)} cols, expected {len(topology)}")
        else:
            for j in range(len(row)):
                if j < len(topology) and i < len(topology[j]):
                    if topology[i][j] != topology[j][i]:
                        issues.append(
                            f"Topology not symmetric at ({i},{j}): "
                            f"{topology[i][j]} != {topology[j][i]}"
                        )

    # Check sensitive hosts exist
    total_hosts = sum(subnets)
    for host_key in sensitive_hosts:
        parsed = _parse_host_key(host_key)
        if parsed is None:
            issues.append(f"Cannot parse sensitive host key: {host_key}")
            continue
        subnet_idx, host_idx = parsed
        if subnet_idx < 0 or subnet_idx >= num_subnets:
            issues.append(f"Sensitive host {host_key}: subnet {subnet_idx} out of range (0-{num_subnets - 1})")
        elif host_idx < 0 or host_idx >= subnets[subnet_idx]:
            issues.append(f"Sensitive host {host_key}: host {host_idx} out of range for subnet {subnet_idx}")

    # Check exploits reference valid services
    for name, exploit in exploits.items():
        svc = exploit.get("service")
        if svc and svc not in services:
            issues.append(f"Exploit '{name}' references unknown service: {svc}")

    # Check privilege escalation references valid processes
    for name, pe in priv_esc.items():
        proc = pe.get("process")
        if proc and proc not in processes:
            issues.append(f"Privilege escalation '{name}' references unknown process: {proc}")

    return ValidationResult(valid=len(issues) == 0, issues=issues)


def _parse_host_key(key: object) -> tuple[int, int] | None:
    """Parse a host key like '(2, 0)' or (2, 0) into a tuple."""
    if isinstance(key, tuple) and len(key) == 2:
        return key
    if isinstance(key, str):
        cleaned = key.strip("() ")
        parts = cleaned.split(",")
        if len(parts) == 2:
            try:
                return int(parts[0].strip()), int(parts[1].strip())
            except ValueError:
                return None
    return None
```

**Step 4: Run tests**

Run: `pytest tests/test_validation.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add cybersim/validation/topology.py tests/test_validation.py
git commit -m "feat: add YAML topology validator (paper Figure 2)"
```

---

### Task 9: Utility Modules (Logger & Helpers)

**Files:**
- Create: `cybersim/utils/logger.py`
- Create: `cybersim/utils/helpers.py`

**Step 1: Write logger.py**

```python
# cybersim/utils/logger.py
"""Logging configuration for cybersim."""

import logging
from pathlib import Path


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Get a configured logger.

    Args:
        name: Logger name.
        log_file: Optional file path for log output.
    """
    logger = logging.getLogger(f"cybersim.{name}")

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
```

**Step 2: Write helpers.py**

```python
# cybersim/utils/helpers.py
"""Shared utility functions."""

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    """Load and validate a YAML file returns a dict."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict from {path}, got {type(data).__name__}")
    return data
```

**Step 3: Commit**

```bash
git add cybersim/utils/
git commit -m "feat: add logger and helper utilities"
```

---

### Task 10: Benchmark Metrics

**Files:**
- Create: `cybersim/benchmarks/metrics.py`
- Test: `tests/test_benchmarks.py`

**Step 1: Write the failing test**

```python
# tests/test_benchmarks.py
import pytest
from cybersim.agents.base import EpisodeResult
from cybersim.benchmarks.metrics import compute_metrics, AgentMetrics


def _make_results(n_success: int, n_fail: int) -> list[EpisodeResult]:
    results = []
    for _ in range(n_success):
        results.append(EpisodeResult(compromised=True, steps=10, total_reward=5.0))
    for _ in range(n_fail):
        results.append(EpisodeResult(compromised=False, steps=50, total_reward=-1.0))
    return results


def test_compute_metrics_all_success():
    results = _make_results(10, 0)
    metrics = compute_metrics(results)
    assert isinstance(metrics, AgentMetrics)
    assert metrics.compromise_rate == 1.0
    assert metrics.avg_steps == 10.0


def test_compute_metrics_mixed():
    results = _make_results(7, 3)
    metrics = compute_metrics(results)
    assert metrics.compromise_rate == pytest.approx(0.7)
    assert metrics.avg_steps == pytest.approx(22.0)


def test_compute_metrics_empty():
    metrics = compute_metrics([])
    assert metrics.compromise_rate == 0.0
    assert metrics.avg_steps == 0.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmarks.py -v`
Expected: FAIL

**Step 3: Write metrics module**

```python
# cybersim/benchmarks/metrics.py
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
```

**Step 4: Run tests**

Run: `pytest tests/test_benchmarks.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add cybersim/benchmarks/metrics.py tests/test_benchmarks.py
git commit -m "feat: add benchmark metric calculations"
```

---

### Task 11: Benchmark Runner

**Files:**
- Create: `cybersim/benchmarks/runner.py`
- Modify: `tests/test_benchmarks.py`

**Step 1: Add failing test**

```python
from cybersim.benchmarks.runner import BenchmarkRunner


def test_benchmark_runner_runs():
    runner = BenchmarkRunner(
        config_paths=["cybersim/configs/baselines/tiny.yaml"],
        num_episodes=2,
        max_steps=50,
    )
    results = runner.run()
    assert len(results) > 0
    # Should have results for at least probing, killchain, stealth
    agent_names = {r.agent_name for r in results}
    assert "probing" in agent_names
    assert "killchain" in agent_names
    assert "stealth" in agent_names
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmarks.py::test_benchmark_runner_runs -v`
Expected: FAIL

**Step 3: Write BenchmarkRunner**

```python
# cybersim/benchmarks/runner.py
"""Benchmark runner: executes all agents across all configs."""

import argparse
import time
from pathlib import Path

from cybersim.agents.base import AttackAgent, EpisodeResult
from cybersim.agents.killchain_agent import KillChainAgent
from cybersim.agents.probing_agent import ProbingAgent
from cybersim.agents.stealth_agent import StealthAgent
from cybersim.benchmarks.metrics import AgentMetrics, compute_metrics
from cybersim.benchmarks.plots import generate_all_plots
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

    # Discover configs if not specified
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

    runner = BenchmarkRunner(
        config_paths=config_paths,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        include_ppo=args.include_ppo,
        ppo_timesteps=args.ppo_timesteps,
    )
    metrics = runner.run()

    # Generate plots
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_all_plots(metrics, output_dir)
    logger.info(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
```

**Step 4: Run test**

Run: `pytest tests/test_benchmarks.py::test_benchmark_runner_runs -v`
Expected: PASS

**Step 5: Commit**

```bash
git add cybersim/benchmarks/runner.py tests/test_benchmarks.py
git commit -m "feat: add BenchmarkRunner with CLI entry point"
```

---

### Task 12: Benchmark Plots

**Files:**
- Create: `cybersim/benchmarks/plots.py`
- Modify: `tests/test_benchmarks.py`

**Step 1: Add failing test**

```python
def test_generate_plots(tmp_path):
    from cybersim.benchmarks.plots import generate_all_plots

    # Create mock metrics
    mock_metrics = [
        AgentMetrics("probing", "tiny", 1.0, 10.0, 2.0, 5.0, 1.0, None, 100),
        AgentMetrics("killchain", "tiny", 0.95, 15.0, 3.0, 4.0, 1.5, None, 100),
        AgentMetrics("stealth", "tiny", 0.8, 28.0, 5.0, 3.0, 2.0, 0.85, 100),
        AgentMetrics("probing", "small", 0.9, 20.0, 4.0, 4.0, 1.0, None, 100),
        AgentMetrics("killchain", "small", 0.85, 25.0, 5.0, 3.5, 1.5, None, 100),
        AgentMetrics("stealth", "small", 0.7, 35.0, 8.0, 2.5, 2.0, 0.78, 100),
    ]

    generate_all_plots(mock_metrics, tmp_path)

    expected_files = [
        "compromise_rates.png",
        "steps_comparison.png",
        "time_comparison.png",
        "scaling.png",
        "agent_radar.png",
        "stealth_tradeoff.png",
    ]
    for filename in expected_files:
        assert (tmp_path / filename).exists(), f"Missing plot: {filename}"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmarks.py::test_generate_plots -v`
Expected: FAIL

**Step 3: Write plots module**

```python
# cybersim/benchmarks/plots.py
"""Benchmark plot generation using matplotlib."""

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from cybersim.benchmarks.metrics import AgentMetrics

# Consistent colours for each agent
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
    """Grouped bar chart: compromise rate per agent x config."""
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
    """Bar chart with error bars: avg steps per agent."""
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
    """Bar chart: average reward as proxy for time efficiency."""
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
    """Line chart: performance vs config complexity."""
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
        speed = 1.0 / max(avg_steps, 1) * 100  # Normalise
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
    """Scatter: stealth score vs compromise rate."""
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
    # Deduplicate legend entries
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
```

**Step 4: Run test**

Run: `pytest tests/test_benchmarks.py::test_generate_plots -v`
Expected: PASS

**Step 5: Commit**

```bash
git add cybersim/benchmarks/plots.py tests/test_benchmarks.py
git commit -m "feat: add benchmark plot generation (6 chart types)"
```

---

### Task 13: Generate YAML Configs via Subagents

**Files:**
- Create: `cybersim/configs/generated/tiny_*.yaml` (3 files)
- Create: `cybersim/configs/generated/small_*.yaml` (3 files)
- Create: `cybersim/configs/generated/medium_*.yaml` (3 files)

**Step 1: Use Claude Code subagents to generate 9 YAML configs**

Launch 3 parallel subagents (one per scale) to generate NASim-compatible YAML configs following the format in `cybersim/configs/baselines/tiny.yaml`. Each config must:
- Follow NASim YAML schema exactly (subnets, topology, sensitive_hosts, os, services, processes, exploits, privilege_escalation, host_configurations, firewall, step_limit)
- Include a comment header: `# LLM-generated configuration - [scale] network`
- Have symmetric topology matrices
- Have consistent host/subnet references

Scale specifications:
- **Tiny** (tiny_corp.yaml, tiny_lab.yaml, tiny_iot.yaml): 1-2 subnets, 3-5 hosts
- **Small** (small_office.yaml, small_campus.yaml, small_cloud.yaml): 3-4 subnets, 8-12 hosts
- **Medium** (medium_enterprise.yaml, medium_datacenter.yaml, medium_hybrid.yaml): 5-8 subnets, 15-25 hosts

**Step 2: Validate all generated configs**

```python
from cybersim.validation.topology import validate_config
from pathlib import Path

for path in sorted(Path("cybersim/configs/generated").glob("*.yaml")):
    result = validate_config(str(path))
    print(f"{path.name}: {'PASS' if result.valid else 'FAIL'}")
    for issue in result.issues:
        print(f"  - {issue}")
```

**Step 3: Fix any validation failures and re-validate**

**Step 4: Commit**

```bash
git add cybersim/configs/generated/
git commit -m "feat: add 9 LLM-generated network configs (tiny/small/medium)"
```

---

### Task 14: README with Benchmark Charts

**Files:**
- Create: `README.md` (replace existing)

**Step 1: Run benchmarks to generate plots**

```bash
python -m cybersim.benchmarks.runner --output assets/ --episodes 100
```

**Step 2: Write README.md**

The README should include:
- Project title and one-line description
- Link to the paper
- Architecture diagram (text-based)
- Installation instructions (`pip install -e ".[dev]"`)
- Quick start (run benchmarks command)
- All 6 benchmark charts embedded as `![](assets/chart_name.png)`
- Agent descriptions table (from the paper)
- Project structure
- Citation block (BibTeX from the paper)
- License

**Step 3: Commit**

```bash
git add README.md assets/
git commit -m "feat: add README with embedded benchmark charts"
```

---

### Task 15: Final Cleanup & Lint

**Files:**
- All `cybersim/` and `tests/` files

**Step 1: Run ruff**

```bash
ruff check cybersim/ tests/ --fix
```

**Step 2: Run all tests**

```bash
pytest tests/ -v --tb=short
```
Expected: ALL PASS

**Step 3: Verify plot generation works end-to-end**

```bash
python -m cybersim.benchmarks.runner --configs cybersim/configs/baselines/tiny.yaml --episodes 10 --output /tmp/test_plots/
ls /tmp/test_plots/
```
Expected: 6 PNG files

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore: lint fixes and final cleanup"
```

---

Plan complete and saved to `docs/plans/2026-03-08-cybersim-implementation.md`. Two execution options:

**1. Subagent-Driven (this session)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** — Open a new session with executing-plans, batch execution with checkpoints

Which approach?