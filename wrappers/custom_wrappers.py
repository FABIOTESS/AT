# wrappers/custom_wrappers.py

import numpy as np
from gym import Wrapper, ActionWrapper

class NumpyToIntActionWrapper(ActionWrapper):
    """
    Action Wrapper to convert actions from NumPy types to Python integers.
    Ensures compatibility with environments that expect actions as Python ints.
    """
    def action(self, action):
        if isinstance(action, np.ndarray):
            action = action.squeeze()
            if action.ndim == 0:
                action = action.item()
        elif isinstance(action, (np.integer,)):
            action = int(action)
        elif not isinstance(action, int):
            try:
                action = int(action)
            except Exception as e:
                raise ValueError(f"Unsupported action type: {type(action)}") from e

        if not isinstance(action, int):
            raise ValueError(f"Action conversion failed. Expected int, got {type(action)}")

        return action

class StepAPICorrector(Wrapper):
    """
    Step API Wrapper to adapt the environment's step function to Gym's API.
    Handles both old (four return values) and new (five return values) step APIs.
    """
    def step(self, action):
        step_result = self.env.step(action)

        if len(step_result) == 4:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
        elif len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            raise ValueError(f"Unexpected number of return values from env.step: {len(step_result)}")

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Adapt the reset method to Gym's API.
        Assumes NASIM's reset returns (obs, info).
        """
        return self.env.reset(**kwargs)
