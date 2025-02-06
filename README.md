# Proving the Utility of LLMs in Cybersecurity Simulations

This is a Python-based toolkit designed for LLM based network simulations. It generates configurations dynamically using Large Language Models to generate YAML files and Pythonic agents.

## Project Overview

This project is a flexible network simulation framework that allows users to define simulation scenarios via YAML configurations. Users can either provide a static YAML file or generate the configuration dynamically using an LLM-powered module. This makes it easy to experiment with different simulation parameters and scenarios.

## Features

- **Dynamic YAML Generation:** Use LLMs to generate simulation configurations YAML on demand.
- **NASim Based:** Is built on top of NASim emulator. It is used to evaluate hallucinations free outputs.
- **Post-Processing:** Automatically normalize configuration blocks to meet simulation requirements.

## How to Run the Project


### LLM-Based Configuration

Run AT/LLM generation/example_llm_generation.ipynb to create YAML configuration files.

### Run The Simulation
Run Main.py to create a benchmark of different approaches (logic behind tthe creation of those is explaine in AT/LLM generation/logic_pythonic_agents.ipynb) vs PPO algorithm.

   
