import os
import gymnasium as gym  # Import gymnasium as gym
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

class StablePPOAgent:
    def __init__(self, config_file, log_dir, total_timesteps, n_eval_episodes):
        """
        Initialize your PPO Agent.
        """
        self.config_file = config_file
        self.log_dir = log_dir
        self.total_timesteps = total_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.model = None

    def load_environment(self):
        """
        Create and return a valid Gymnasium environment wrapped in Monitor.
        """
        env = gym.make("CartPole-v1")  # Use gymnasium
        print(f"DEBUG: Created environment of type: {type(env)}")  # Debug statement
        env = Monitor(env)  # Monitor expects a single Env
        print(f"DEBUG: Wrapped environment in Monitor: {type(env)}")  # Debug statement
        return env

    def train(self):
        """
        Train the PPO model on the environment.
        """
        train_env = self.load_environment()

        # Create the model
        self.model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log=os.path.join(self.log_dir, "tb_logs")
        )

        # Train the model
        self.model.learn(total_timesteps=self.total_timesteps)

        # Cleanup
        train_env.close()

    def evaluate(self):
        """
        Evaluate the trained model.
        Returns: List of booleans indicating success per episode.
        """
        if self.model is None:
            raise ValueError("Model not found. Please call train() first.")

        eval_env = self.load_environment()
        successes = []

        for _ in range(self.n_eval_episodes):
            obs, info = eval_env.reset()
            done = False
            truncated = False

            while not (done or truncated):
                action, _states = self.model.predict(obs)
                obs, reward, done, truncated, info = eval_env.step(action)
                # Customize your success condition as needed
            # Example success condition: episode ended with a positive reward
            successes.append(reward > 0)

        eval_env.close()
        return successes

    def train_and_evaluate(self):
        """
        Train the model and evaluate in sequence.
        Return a DataFrame so `.to_csv(...)` works in approach0.py
        """
        self.train()
        results_list = self.evaluate()  # e.g., [True, False, True, ...]

        # Convert the list of booleans into a Pandas DataFrame
        df = pd.DataFrame({
            "success": results_list
        })
        return df
