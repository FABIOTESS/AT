# environments/environment_loader.py

import os
import yaml
import nasim
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wrappers.custom_wrappers import NumpyToIntActionWrapper, StepAPICorrector
from utils.helpers import load_yaml_config

def load_environment(config_file):
    """
    Loads the NASIM environment with the specified configuration file and applies necessary wrappers.

    Parameters:
    - config_file (str): Path to the NASIM network configuration YAML file.

    Returns:
    - env (gym.Env): Wrapped environment ready for training.
    """
    with open(config_file, 'r') as file:
        network_config = load_yaml_config(file)

    env = nasim.load(
        config_file,
        fully_obs=True,
        flat_actions=True,
        flat_obs=True,
        # Add other necessary parameters from network_config if needed
    )

    env = NumpyToIntActionWrapper(env)
    env = StepAPICorrector(env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")

    return env
