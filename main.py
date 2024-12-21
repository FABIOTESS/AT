# main.py

import os
import pandas as pd
from simulations.approach0 import run_ppo_simulation
from simulations.approach1 import run_approach1
from simulations.approach2 import run_approach2
from simulations.approach3 import run_approach3
from utils.helpers import setup_logger
from datetime import datetime


def write_results_to_csv_pandas(all_results, filename='simulation_report.csv'):
    """
    Writes all aggregated simulation results to a CSV file using pandas.

    Parameters:
    - all_results (list of dict): List of dictionaries containing results from all simulation approaches across iterations.
    - filename (str): Name of the CSV file to create.
    """
    # Define the CSV file path
    csv_file_path = os.path.join('reports', filename)

    # Ensure the 'reports' directory exists
    os.makedirs('reports', exist_ok=True)

    # Create a DataFrame
    df = pd.DataFrame(all_results, columns=[
        'Iteration', 'Datestamp', 'Approach', 'Total Runs',
        'Successful Attacks', 'Unsuccessful Attacks',
        'Total Time Taken (s)', 'Average Time per Run (s)'
    ])

    try:
        # Write the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False)
        print(f"Simulation report successfully saved to {csv_file_path}")

    except Exception as e:
        print(f"Failed to write simulation report to CSV using pandas: {e}")


def main(config_file='config/config.yaml', main_log_dir='logs', master_number=10):
    """
    Main function to run all simulation approaches.

    Returns:
    - aggregate_results (dict): Dictionary containing results from all simulation approaches.
    """
    # Ensure the main log directory exists
    os.makedirs(main_log_dir, exist_ok=True)
    main_logger = setup_logger('main', os.path.join(main_log_dir, 'main.log'))

    main_logger.info("Starting all simulation approaches.")

    # Approach 0: PPO-Based Simulation
    approach0_log_dir = os.path.join(main_log_dir, 'approach0_logs')
    approach0_results = run_ppo_simulation(
        master_number=master_number,
        config_file=config_file,
        log_dir=approach0_log_dir
    )

    # Approach 1: Manual Attack Simulation
    approach1_log_dir = os.path.join(main_log_dir, 'approach1_logs')
    approach1_results = run_approach1(
        master_number=master_number,
        config_file=config_file,
        log_dir=approach1_log_dir
    )

    # Approach 2: Cyber Kill Chain Simulation
    approach2_log_dir = os.path.join(main_log_dir, 'approach2_logs')
    approach2_results = run_approach2(
        master_number=master_number,
        config_file=config_file,
        log_dir=approach2_log_dir
    )

    # Approach 3: Privilege Escalation Simulation
    approach3_log_dir = os.path.join(main_log_dir, 'approach3_logs')
    approach3_results = run_approach3(
        master_number=master_number,
        config_file=config_file,
        log_dir=approach3_log_dir
    )

    # Aggregate all results
    aggregate_results = {
        'Approach 0 (PPO-Based)': approach0_results,
        'Approach 1 (Manual Attack)': approach1_results,
        'Approach 2 (Cyber Kill Chain Simulation)': approach2_results,
        'Approach 3 (Privilege Escalation)': approach3_results
    }

    # Log aggregate results
    main_logger.info("All simulation approaches completed.")
    for approach, result in aggregate_results.items():
        if result:
            main_logger.info(f"\n{approach} Results:")
            for key, value in result.items():
                main_logger.info(f"{key}: {value}")
        else:
            main_logger.warning(f"{approach} did not return any results.")

    # Print aggregate results to console
    print("\n======================================")
    print("All Simulation Approaches Experiment Summary")
    print("======================================")
    for approach, result in aggregate_results.items():
        if result:
            print(f"\n{approach} Results:")
            for key, value in result.items():
                print(f"{key}: {value}")
        else:
            print(f"\n{approach} did not return any results.")

    return aggregate_results


if __name__ == "__main__":
    all_results = []
    i = 0
    while i < 6:
        # Capture the current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Run the main simulation with the specific config file
        aggregate_results = main(config_file=f'config/config{i}.yaml', main_log_dir='logs', master_number=100)

        # Iterate through each approach's results and append to all_results with iteration and timestamp
        for approach, result in aggregate_results.items():
            if result:
                row = {
                    'Iteration': i,
                    'Datestamp': timestamp,
                    'Approach': approach,
                    'Total Runs': result.get('Total Runs', 0),
                    'Successful Attacks': result.get('Successful Attacks', 0),
                    'Unsuccessful Attacks': result.get('Unsuccessful Attacks', 0),
                    'Total Time Taken (s)': round(result.get('Total Time Taken', 0), 4),
                    'Average Time per Run (s)': round(result.get('Average Time per Run', 0), 4)
                }
            else:
                row = {
                    'Iteration': i,
                    'Datestamp': timestamp,
                    'Approach': approach,
                    'Total Runs': 'N/A',
                    'Successful Attacks': 'N/A',
                    'Unsuccessful Attacks': 'N/A',
                    'Total Time Taken (s)': 'N/A',
                    'Average Time per Run (s)': 'N/A'
                }
            all_results.append(row)

        i += 1

    # Write all collected results to the CSV file
    write_results_to_csv_pandas(all_results)
