# simulations/approach3.py

import os
import yaml
import time
import random
from tqdm import tqdm
from utils.helpers import load_yaml_config, setup_logger

def run_approach3(master_number=1000, config_file='config/config.yaml', log_dir='approach3_logs'):
    """
    Runs Approach 3 simulation multiple times.

    Parameters:
    - master_number (int): Number of simulation runs.
    - config_file (str): Path to the main configuration YAML file.
    - log_dir (str): Directory to save logs.

    Returns:
    - results (dict): Dictionary containing success and failure counts and timing information.
    """
    # Setup logger
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger('approach3', os.path.join(log_dir, 'approach3.log'))

    test_results = []
    successful_attacks = []
    unsuccessful_attacks = []
    time_taken = []

    # Load network configuration
    try:
        main_config = load_yaml_config(config_file)
        scenario_file = main_config.get('network_config_file')

        if not scenario_file:
            logger.error("network_config_file not specified in config.yaml")
            raise ValueError("network_config_file not specified in config.yaml")

        # Load the scenario configuration
        network_config = load_yaml_config(scenario_file)
        logger.info(f"Loaded network configuration from {scenario_file}")
    except Exception as e:
        logger.error(f"Failed to load network configuration: {e}")
        return {}

    # Extract configuration details
    host_configurations = network_config.get('host_configurations', {})
    exploits = network_config.get('exploits', {})
    privilege_escalation = network_config.get('privilege_escalation', {})
    sensitive_hosts = network_config.get('sensitive_hosts', [])

    def simulate_attack():
        """
        Simulates a single attack scenario.

        Returns:
        - compromised (bool): True if network was compromised, False otherwise.
        """
        network_map = {}
        attack_log = []

        # Initial compromised hosts
        initial_hosts = ['(1, 0)', '(2, 0)']  # Adjust based on actual host identifiers
        for host in initial_hosts:
            network_map[host] = {'compromised': True, 'access_level': 'user'}
            attack_log.append(f"Initial compromise: {host}")
            logger.debug(f"Host {host} compromised with 'user' access.")

        initial_access = []

        # Start with scanning the network and attempting exploits
        for host, config in host_configurations.items():
            for service in config.get('services', []):
                for exploit_name, exploit in exploits.items():
                    if exploit.get('service') == service and exploit.get('os') == config.get('os', ''):
                        if random.random() <= exploit.get('prob', 0):
                            initial_access.append((host, exploit.get('access', 'user')))
                            attack_log.append(f"Exploit {exploit_name} successful on {host}")
                            logger.info(f"Exploit {exploit_name} successful on {host}")
                            break  # Assuming one exploit per service per host

        # Attempt privilege escalation based on initial access
        for access in initial_access:
            host, current_access = access
            if current_access == 'user':
                for pe_name, pe in privilege_escalation.items():
                    if (pe.get('process') in host_configurations.get(host, {}).get('processes', []) and
                        pe.get('os') == host_configurations.get(host, {}).get('os', '')):
                        if random.random() <= pe.get('prob', 0):
                            network_map[host]['access_level'] = 'root'
                            attack_log.append(f"Privilege escalation {pe_name} successful on {host}")
                            logger.info(f"Privilege escalation {pe_name} successful on {host}")
                            break  # Assuming one privilege escalation per host

            # Check if sensitive host is compromised
            if network_map.get(host, {}).get('access_level') == 'root' and host in sensitive_hosts:
                attack_log.append("Network was compromised")
                logger.info(f"Sensitive host {host} compromised.")
                return True

        # After all attempts
        compromised_hosts = [h for h, status in network_map.items() if status.get('compromised', False)]
        if any(host in sensitive_hosts for host in compromised_hosts):
            attack_log.append("Network was compromised")
            logger.info("Network was compromised.")
            return True
        else:
            attack_log.append("Network was not compromised")
            logger.info("Network was not compromised.")
            return False

    # Run simulations
    for run in tqdm(range(1, master_number + 1), desc="Running Approach 3 Simulations"):
        logger.info(f"Starting Run {run}/{master_number}")
        start_run_time = time.time()

        try:
            success = simulate_attack()

            if success:
                test_results.append(1)
                successful_attacks.append(1)
                unsuccessful_attacks.append(0)
                logger.debug(f"Run {run}: Successful Attack")
            else:
                test_results.append(0)
                successful_attacks.append(0)
                unsuccessful_attacks.append(1)
                logger.debug(f"Run {run}: Unsuccessful Attack")

            elapsed_time = time.time() - start_run_time
            time_taken.append(elapsed_time)

            logger.info(f"Run {run} Completed - Success: {success}, Time Taken: {elapsed_time:.2f} seconds")
            print(f"Run {run} Completed - Success: {success}, Time Taken: {elapsed_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Run {run} failed: {e}")
            test_results.append(0)
            unsuccessful_attacks.append(1)
            successful_attacks.append(0)
            elapsed_time = time.time() - start_run_time
            time_taken.append(elapsed_time)
            print(f"Run {run} Failed - Error: {e}, Time Taken: {elapsed_time:.2f} seconds")

    # Aggregate results
    total_success = sum(successful_attacks)
    total_unsuccessful = sum(unsuccessful_attacks)
    total_time = sum(time_taken)
    average_time = total_time / master_number if master_number > 0 else 0

    results = {
        'Total Runs': master_number,
        'Successful Attacks': total_success,
        'Unsuccessful Attacks': total_unsuccessful,
        'Total Time Taken': total_time,
        'Average Time per Run': average_time
    }

    # Log summary
    logger.info("\n======================================")
    logger.info("Approach 3 Experiment Summary")
    logger.info("======================================")
    for key, value in results.items():
        logger.info(f"{key}: {value}")

    # Print summary to console
    print("\n======================================")
    print("Approach 3 Experiment Summary")
    print("======================================")
    for key, value in results.items():
        print(f"{key}: {value}")

    return results
