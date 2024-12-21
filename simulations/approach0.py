import os
import time
import traceback
from tqdm import tqdm
import pandas as pd
from agents.ppo_agent import StablePPOAgent
from utils.helpers import load_yaml_config, setup_logger


def run_ppo_simulation(master_number=10, config_file='config/config.yaml', log_dir='approach0_logs'):
    """
    Runs the PPO-based simulation approach multiple times, but:
      - Trains the PPO agent only once outside the main loop.
      - Measures only the evaluation (inference) time for each run.

    Parameters:
        master_number (int): Number of evaluation runs (NOT training runs).
        config_file (str): Path to the main configuration YAML file.
        log_dir (str): Directory to save logs and models.

    Returns:
        results (dict): Dictionary containing success/failure stats and timing.
    """
    # Ensure the logging directory exists
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger('approach0', os.path.join(log_dir, 'approach0.log'))

    test_results = []
    successful_attacks = []
    unsuccessful_attacks = []
    time_taken = []

    # Initialize the PPO agent
    agent = StablePPOAgent(
        config_file=config_file,
        log_dir=log_dir,
        total_timesteps=100000,
        n_eval_episodes=1  # We'll evaluate once per run, but agent is trained once below
    )

    # 1) Train the agent once (ignore this time for the "time_taken" metric)
    logger.info("Training PPO agent once, ignoring training time for subsequent calculations...")
    agent.train()

    # Path to the CSV file (if you want to save results per run)
    file_path = '/Users/fabio/PycharmProjects/PythonProject/simulations/test.csv'

    # 2) Evaluate multiple times, measuring only evaluation (inference) time
    for run in tqdm(range(1, master_number + 1), desc="Running PPO Experiments"):
        logger.info(f"Starting Evaluation Run {run}/{master_number}")

        # Start timing *only* the evaluation
        start_run_time = time.time()

        try:
            # This call should only perform inference/evaluation, not training
            state = agent.evaluate()  # <-- Replace with your agent's evaluation method

            logger.debug(f"Run {run} - Evaluation State: {state}")

            # Check whether state is a DataFrame with a 'success' column, a bool, or something else
            success_value = False
            if isinstance(state, pd.DataFrame) and 'success' in state.columns and not state.empty:
                success_value = bool(state['success'].iloc[0])
            elif isinstance(state, bool):
                success_value = state

            if success_value:
                test_results.append(1)
                successful_attacks.append(1)
                unsuccessful_attacks.append(0)
            else:
                test_results.append(0)
                successful_attacks.append(0)
                unsuccessful_attacks.append(1)

            elapsed_time = time.time() - start_run_time
            time_taken.append(elapsed_time)

            print(
                f"Run {run} Completed - Success: {success_value}, "
                f"Evaluation Time: {elapsed_time:.2f} seconds"
            )

            # (Optional) If you want to store run-by-run results in a CSV:
            run_data = pd.DataFrame({
                "Run": [run],
                "Success": [success_value],
                "EvaluationTime_s": [elapsed_time]
            })
            if not os.path.exists(file_path):
                run_data.to_csv(file_path, index=False)
            else:
                run_data.to_csv(file_path, mode="a", header=False, index=False)

        except Exception as e:
            logger.error(f"Run {run} - Exception occurred: {e}")
            logger.error(traceback.format_exc())
            test_results.append(0)
            successful_attacks.append(0)
            unsuccessful_attacks.append(1)

            elapsed_time = time.time() - start_run_time
            time_taken.append(elapsed_time)
            print(
                f"Run {run} Failed - Exception: {e}, "
                f"Evaluation Time: {elapsed_time:.2f} seconds"
            )

    # Aggregate results (only for the evaluation)
    total_success = sum(successful_attacks)
    total_unsuccessful = sum(unsuccessful_attacks)
    total_eval_time = sum(time_taken)
    average_eval_time = (total_eval_time / master_number) if master_number > 0 else 0

    results = {
        'Total Runs': master_number,
        'Successful Attacks': total_success,
        'Unsuccessful Attacks': total_unsuccessful,
        'Total Evaluation Time (s)': total_eval_time,
        'Average Evaluation Time per Run (s)': average_eval_time
    }

    # Log summary
    logger.info("\n======================================")
    logger.info("Approach 0 Experiment Summary")
    logger.info("======================================")
    for key, value in results.items():
        logger.info(f"{key}: {value}")

    # Print summary to console
    print("\n======================================")
    print("Approach 0 Experiment Summary")
    print("======================================")
    for key, value in results.items():
        print(f"{key}: {value}")

    return results


if __name__ == "__main__":
    # Example usage with just 1 run
    run_ppo_simulation(master_number=1)
