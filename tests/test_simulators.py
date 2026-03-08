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
