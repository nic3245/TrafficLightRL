# Traffic Signal Control with Reinforcement Learning

This project implements and evaluates reinforcement learning agents for traffic signal control using the CityFlow traffic simulator. It compares the performance of different agent architectures (DQN and Linear) under various configurations.

## Project Overview

This repository contains a framework for training and evaluating reinforcement learning agents to optimize traffic signal timing. The main components are:

- **Base Agent**: Abstract class that defines the interface for all agents
- **DQN Agent**: Deep Q-Network implementation with PyTorch
- **Linear Agent**: Linear function approximation for Q-learning
- **Simulator**: Wrapper around CityFlow for reinforcement learning environment
- **Training**: Functions to train agents and track performance metrics
- **Experiments**: Suite of experiments to compare agents and hyperparameters

## Setup

### Docker Setup

The project uses the CityFlow simulator which is available as a Docker image:

```bash
# Pull the CityFlow Docker image
docker pull cityflowproject/cityflow:latest

# Start container with your local directory mounted
docker run -it -v "/path/to/your/project:/workspace" cityflowproject/cityflow:latest

# Navigate to the mounted directory
cd workspace
```

### Dependencies

The project requires the following Python packages:

```
torch
numpy
matplotlib
pandas
```

You can install them using:

```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

The main entry point is `run_experiments.py`, which provides several experiment options:

```bash
# Run all experiments (comprehensive evaluation)
python run_experiments.py --experiment comprehensive

# Compare DQN vs. Linear agents
python run_experiments.py --experiment agent_comparison

# Run hyperparameter sweep
python run_experiments.py --experiment hyperparam_sweep --agent dqn

# Compare action space configurations
python run_experiments.py --experiment action_space --agent dqn

# Train a single model
python run_experiments.py --experiment single_model --agent dqn --episodes 50
```

### Key Parameters

- `--agent`: Agent type to use (`dqn` or `linear`)
- `--episodes`: Number of episodes to train for
- `--experiment`: Type of experiment to run

## Project Structure

- `base_agent.py`: Abstract base class for all RL agents
- `dqn_model.py`: DQN agent implementation
- `linear_approximator.py`: Linear function approximation agent
- `simulation.py`: CityFlow simulator wrapper
- `train.py`: Training loop and utilities
- `run_experiments.py`: Experiment configurations and analysis

## Results

Experiment results are saved in the `results/` directory with the following structure:

- CSV files for metrics (rewards, epsilon, loss, etc.)
- Summary statistics
- Performance plots
- JSON configuration records

## Models

Trained models are saved in `models_dqn_actions{N}/` or `models_linear_actions{N}/` directories:

- `best_model.pth/.npz`: Model with highest reward
- `checkpoint_episode_{N}.pth/.npz`: Regular checkpoints
- `final_model.pth/.npz`: Final model after training

This folder also stores training plots/curves.

## Frontend
After running a model, its replay will be stored in the config folder as `replay.txt` and `replay_roadnet.json`. By going into the frontend folder and opening `index.html` in your browser, you can load these files and view the replay.