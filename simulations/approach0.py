import os
import time
import traceback
from tqdm import tqdm
import pandas as pd  # Ensure pandas is imported
from agents.ppo_agent import StablePPOAgent
from utils.helpers import load_yaml_config, setup_logger

def run_ppo_simulation(master_number=10, config_file='config/config.yaml', log_dir='approach0_logs'):
    """
    Runs the PPO-based simulation approach multiple times.

    Parameters:
        master_number (int): Number of simulation runs.
        config_file (str): Path to the main configuration YAML file.
        log_dir (str): Directory to save logs and models.

    Returns:
        results (dict): Dictionary containing success/failure stats and timing.
    """
    # Setup logger and ensure log_dir exists
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
        n_eval_episodes=1  # Evaluate per run
    )

    # Where we save the combined data
    file_path = '/Users/fabio/PycharmProjects/PythonProject/simulations/test.csv'

    for run in tqdm(range(1, master_number + 1), desc="Running PPO Experiments"):
        logger.info(f"Starting Run {run}/{master_number}")
        start_run_time = time.time()

        try:
            state = agent.train_and_evaluate()  # Expected to be a DataFrame with a "success" column

            # Debug: Log the state to verify its structure
            logger.debug(f"Run {run} - Evaluation State: {state}")

            if isinstance(state, pd.DataFrame):
                # Assuming there's a 'success' column and a single row
                success_value = state['success'].iloc[0]
                if success_value:
                    test_results.append(1)
                    successful_attacks.append(1)
                    unsuccessful_attacks.append(0)
                else:
                    test_results.append(0)
                    successful_attacks.append(0)
                    unsuccessful_attacks.append(1)
            elif isinstance(state, bool):
                # If train_and_evaluate() directly returns a boolean
                if state:
                    test_results.append(1)
                    successful_attacks.append(1)
                    unsuccessful_attacks.append(0)
                else:
                    test_results.append(0)
                    successful_attacks.append(0)
                    unsuccessful_attacks.append(1)
            else:
                logger.warning(f"Run {run} - Unexpected state type: {type(state)}. Treating as failure.")
                test_results.append(0)
                successful_attacks.append(0)
                unsuccessful_attacks.append(1)

            elapsed_time = time.time() - start_run_time
            time_taken.append(elapsed_time)

            print(
                f"Run {run} Completed - Success: {success_value if isinstance(state, pd.DataFrame) else state}, "
                f"Time Taken: {elapsed_time:.2f} seconds"
            )

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
                f"Time Taken: {elapsed_time:.2f} seconds"
            )

    # Aggregate results
    total_success = sum(successful_attacks)
    total_unsuccessful = sum(unsuccessful_attacks)
    total_time = sum(time_taken)
    average_time = total_time / master_number if master_number > 0 else 0

    results = {
        'Total Runs': master_number,
        'Successful Attacks': total_success,
        'Unsuccessful Attacks': total_unsuccessful,
        'Total Time Taken (s)': total_time,
        'Average Time per Run (s)': average_time
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
    # Example usage
    run_ppo_simulation(master_number=1)
