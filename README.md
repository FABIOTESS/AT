# AT: Advanced Network Simulation Toolkit

AT is a Python-based toolkit designed for advanced network simulations. It offers flexible YAML configuration options, including the ability to generate configurations dynamically using Large Language Models (LLMs) or by loading static YAML files. This README provides step-by-step instructions on how to set up, configure, and run the project.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [How to Run the Project](#how-to-run-the-project)
  - [Using File-Based Configuration](#using-file-based-configuration)
  - [Using LLM-Based Configuration](#using-llm-based-configuration)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The AT project is a flexible network simulation framework that allows users to define simulation scenarios via YAML configurations. Users can either provide a static YAML file or generate the configuration dynamically using an LLM-powered module. This makes it easy to experiment with different simulation parameters and scenarios.

## Features

- **Dynamic YAML Generation:** Use LLMs to generate simulation configurations on demand.
- **Flexible Configuration:** Choose between static YAML files or dynamic LLM-based configuration.
- **CLI Interface:** Run simulations using a straightforward command-line interface.
- **Post-Processing:** Automatically normalize configuration blocks to meet simulation requirements.

## Repository Structure

```
AT/
├── enhanced_yaml/
│   ├── __init__.py
│   ├── enhanced_yaml.py        # LLM integration and YAML generation code
│   └── ...
├── at_runner.py                # Main entry point for running simulations
├── loader.py                   # Handles configuration loading and post-processing
├── yaml_llm.py                 # Adapter for LLM-based YAML generation
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation (this file)
```

## Prerequisites

Before running the project, ensure you have the following installed:

- **Python 3.6+**
- **pip** (Python package installer)

For an isolated environment, you can set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Setup and Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/FABIOTESS/AT.git
   cd AT
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**

   If you plan to use LLM-based configuration, set your API key. For example:

   ```bash
   export OPENAI_API_KEY="YOUR_API_KEY"
   ```

## How to Run the Project

The project is executed from the command line. The main script, `at_runner.py`, supports two configuration modes: file-based and LLM-based.

### Using File-Based Configuration

1. **Create a YAML Configuration File:**

   Create a file (e.g., `config.yaml`) with your simulation parameters:

   ```yaml
   simulation:
     scenario_name: "LocalTest"
     network_simulation:
       timeout: 30
       nodes: 5
   ```

2. **Run the Simulation:**

   ```bash
   python at_runner.py --config-source file --config-file config.yaml
   ```

### Using LLM-Based Configuration

1. **Provide Simulation Parameters via Command-Line:**

   The parameters you pass will be used by the LLM to generate a YAML configuration. For example:

   ```bash
   python at_runner.py --config-source llm --param1 "example_value" --param2 "another_value"
   ```

   Under the hood, `at_runner.py` collects these parameters, calls the LLM through `yaml_llm.py`, and then parses the returned YAML for simulation setup.

## Examples

Below are some examples to help you get started:

- **File-Based Example:**

   1. Create `config.yaml` with the following content:

      ```yaml
      simulation:
        scenario_name: "LocalTest"
        network_simulation:
          timeout: 30
          nodes: 5
      ```

   2. Run:

      ```bash
      python at_runner.py --config-source file --config-file config.yaml
      ```

- **LLM-Based Example:**

   Run:

      ```bash
      python at_runner.py --config-source llm --param1 "simulation_param_value" --param2 "another_value"
      ```

## Contributing

Contributions are welcome! To contribute:

1. **Fork** the repository.
2. **Create** a new branch for your feature or bug fix:

   ```bash
   git checkout -b feature/my-improvement
   ```

3. **Commit** your changes with clear, descriptive messages.
4. **Push** your branch to your fork:

   ```bash
   git push origin feature/my-improvement
   ```

5. **Open** a Pull Request with a detailed explanation of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Happy simulating! If you encounter any issues or have questions, please open an issue or submit a pull request.
