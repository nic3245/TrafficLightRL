# Main driver
import numpy as np
import random
import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
from train import train_model

def export_results_to_csv(results, experiment_name="experiment"):
    """
    Export training results to CSV files for easy import into Word
    
    Args:
        results: Tuple of training metrics (rewards, epsilons, losses, q_values, waiting_vehicles, phase_changes)
        experiment_name: Prefix for saved files
    """
    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Unpack results
    rewards, epsilons, losses, q_values, waiting_vehicles, phase_changes = results
    
    # Create episode numbers
    episodes = list(range(1, len(rewards) + 1))
    
    # Create DataFrames for each metric
    rewards_df = pd.DataFrame({"Episode": episodes, "Reward": rewards})
    epsilon_df = pd.DataFrame({"Episode": episodes, "Epsilon": epsilons})
    loss_df = pd.DataFrame({"Episode": episodes, "Loss": losses})
    q_values_df = pd.DataFrame({"Episode": episodes, "Average Q-Value": q_values})
    waiting_df = pd.DataFrame({"Episode": episodes, "Average Waiting Vehicles": waiting_vehicles})
    phase_changes_df = pd.DataFrame({"Episode": episodes, "Phase Changes": phase_changes})
    
    # Save each metric to CSV
    rewards_df.to_csv(f"{results_dir}/{experiment_name}_rewards.csv", index=False)
    epsilon_df.to_csv(f"{results_dir}/{experiment_name}_epsilon.csv", index=False)
    loss_df.to_csv(f"{results_dir}/{experiment_name}_loss.csv", index=False)
    q_values_df.to_csv(f"{results_dir}/{experiment_name}_q_values.csv", index=False)
    waiting_df.to_csv(f"{results_dir}/{experiment_name}_waiting.csv", index=False)
    phase_changes_df.to_csv(f"{results_dir}/{experiment_name}_phase_changes.csv", index=False)
    
    # Also save a combined summary table for final performance
    summary_data = {
        "Metric": ["Total Reward", "Average Q-Value", "Average Waiting Vehicles", "Phase Changes"],
        "Final Value": [rewards[-1], q_values[-1], waiting_vehicles[-1], phase_changes[-1]],
        "Average (Last 5 Episodes)": [
            np.mean(rewards[-5:]), 
            np.mean(q_values[-5:]), 
            np.mean(waiting_vehicles[-5:]), 
            np.mean(phase_changes[-5:])
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{results_dir}/{experiment_name}_summary.csv", index=False)
    
    print(f"Results exported to CSV files in {results_dir}/ with prefix {experiment_name}")

def run_experiment(experiment_name, agent_type='dqn', episodes=50, steps_per_episode=1000, 
                  hyperparams=None, action_space_size=5, random_seed=None):
    """
    Run a complete experiment with proper result storage and reporting
    
    Args:
        experiment_name: Name for this experiment run
        agent_type: 'dqn' or 'linear'
        episodes: Number of episodes to train
        steps_per_episode: Steps per episode
        hyperparams: Dictionary of hyperparameters to override defaults
        action_space_size: Size of action space (4 or 5)
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with experiment results
    """
    # Set random seeds for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
    
    print(f"Starting experiment: {experiment_name}")
    print(f"Agent: {agent_type}, Episodes: {episodes}, Steps: {steps_per_episode}")
    print(f"Action space size: {action_space_size}")
    
    # Create results directory
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    experiment_dir = f"{results_dir}/{experiment_name}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
        
    # Train the agent
    results = train_model(
        agent_type=agent_type,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        debug=True,
        save_checkpoint=True,
        action_space_size=action_space_size,
        hyperparams=hyperparams
    )
    
    # Unpack results
    rewards, epsilons, losses, q_values, waiting_vehicles, phase_changes = results
    
    # Export to CSV for further analysis
    export_results_to_csv(results, f"{experiment_name}_{agent_type}")
    
    # Calculate summary statistics
    final_reward = float(rewards[-1])
    final_waiting = float(waiting_vehicles[-1])
    avg_reward_last5 = float(np.mean(rewards[-5:]))
    avg_waiting_last5 = float(np.mean(waiting_vehicles[-5:]))
    
    # Store experiment configuration
    experiment_config = {
        "name": experiment_name,
        "agent_type": agent_type,
        "episodes": episodes,
        "steps_per_episode": steps_per_episode,
        "action_space_size": action_space_size,
        "random_seed": random_seed,
        "hyperparameters": hyperparams
    }
    
    # Convert numpy arrays to Python lists with float conversion for JSON
    rewards_list = [float(x) for x in rewards]
    epsilons_list = [float(x) for x in epsilons]
    losses_list = [float(x) for x in losses]
    q_values_list = [float(x) for x in q_values]
    waiting_vehicles_list = [float(x) for x in waiting_vehicles]
    phase_changes_list = [int(x) for x in phase_changes]
    
    # Create a summary dictionary
    summary = {
        "config": experiment_config,
        "final_reward": final_reward,
        "final_waiting_vehicles": final_waiting,
        "avg_reward_last5": avg_reward_last5,
        "avg_waiting_vehicles_last5": avg_waiting_last5,
        "rewards": rewards_list,
        "epsilons": epsilons_list,
        "losses": losses_list,
        "q_values": q_values_list,
        "waiting_vehicles": waiting_vehicles_list,
        "phase_changes": phase_changes_list
    }
    
    # Save summary as JSON
    import json
    with open(f"{experiment_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Experiment '{experiment_name}' completed successfully")
    print(f"Results saved to {experiment_dir}/")
    
    return summary

def run_agent_comparison(episodes=50):
    dqn_summary = run_experiment(
        experiment_name="agent_comparison/dqn",
        agent_type="dqn",
        episodes=episodes,
        steps_per_episode=1000,
        action_space_size=5,
        random_seed=42
    )
    
    linear_summary = run_experiment(
        experiment_name="agent_comparison/linear",
        agent_type="linear",
        episodes=episodes,
        steps_per_episode=1000,
        action_space_size=5,
        random_seed=42
    )
    
    # Compare DQN vs Linear directly
    plt.figure(figsize=(10, 6))
    plt.plot(dqn_summary['rewards'], label='DQN')
    plt.plot(linear_summary['rewards'], label='Linear')
    plt.title("DQN vs Linear Agent Performance")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/agent_comparison/comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def run_hyperparameter_sweep(experiment_name, agent_type='dqn', param_name='learning_rate', 
                           param_values=[0.001, 0.0005, 0.0001], 
                           episodes=30, steps_per_episode=1000, action_space_size=5):
    """
    Run a sweep over a single hyperparameter and compare results
    """
    results = {}
    
    # Create experiment directory
    base_dir = f"results/{experiment_name}"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Run experiment for each parameter value
    for value in param_values:
        # Create hyperparameter dictionary
        hyperparams = {param_name: value}
        
        # Name for this specific run
        run_name = f"{param_name}_{value}"
        
        print(f"\n{'-' * 80}")
        print(f"Running {experiment_name} with {param_name}={value}")
        print(f"{'-' * 80}\n")
        
        # Run experiment
        summary = run_experiment(
            experiment_name=f"{experiment_name}/{run_name}",
            agent_type=agent_type,
            episodes=episodes,
            steps_per_episode=steps_per_episode,
            hyperparams=hyperparams,
            action_space_size=action_space_size
        )
        
        # Store results
        results[value] = summary
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    
    for value, summary in results.items():
        plt.plot(summary['rewards'], label=f"{param_name}={value}")
    
    plt.title(f"Impact of {param_name} on {agent_type.upper()} Agent Performance")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    
    # Save the comparison plot
    comparison_path = f"{base_dir}/comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary CSV for further analysis
    summary_data = []
    
    for value, summary in results.items():
        summary_data.append({
            param_name: value,
            "Final Reward": summary["final_reward"],
            "Avg Reward (Last 5)": summary["avg_reward_last5"],
            "Final Waiting Vehicles": summary["final_waiting_vehicles"],
            "Avg Waiting Vehicles (Last 5)": summary["avg_waiting_vehicles_last5"]
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{base_dir}/comparison_summary.csv", index=False)
    
    print(f"Hyperparameter sweep for {param_name} completed")
    print(f"Summary saved to {base_dir}/comparison_summary.csv")
    
    return results

def run_action_space(agent_type='dqn', episodes=40):
    action4_summary = run_experiment(
        experiment_name="action_space_comparison/action4",
        agent_type=agent_type,
        action_space_size=4,
        episodes=episodes
    )
    
    action5_summary = run_experiment(
        experiment_name="action_space_comparison/action5",
        agent_type=agent_type,
        action_space_size=5,
        episodes=episodes
    )
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    plt.plot(action4_summary['rewards'], label='4-Action Space')
    plt.plot(action5_summary['rewards'], label='5-Action Space')
    plt.title("Impact of Action Space Size on Performance")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/action_space_comparison/comparison.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run reinforcement learning experiments for traffic control')
    parser.add_argument('--experiment', choices=['comprehensive', 'agent_comparison', 'hyperparam_sweep', 'action_space', 'single_model'], 
                        default='comprehensive', help='Which experiment to run')
    parser.add_argument('--agent', choices=['dqn', 'linear'], default='dqn', help='Agent type for individual experiments')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes to train for individual experiments')
    
    args = parser.parse_args()
    
    if args.experiment == 'comprehensive':
        # -------------------- 1. Agent Comparison (DQN vs Linear) -------------
        run_agent_comparison()
        
        # -------------------- 2. Hyperparameter Sensitivity Analysis ----------

        run_hyperparameter_sweep(
            experiment_name="hyperparameter_sweep/learning_rate_dqn",
            agent_type="dqn",
            param_name="learning_rate",
            param_values=[0.001, 0.0005, 0.0001],
            episodes=30
        )
        
        run_hyperparameter_sweep(
            experiment_name="hyperparameter_sweep/epsilon_decay_dqn",
            agent_type="dqn",
            param_name="epsilon_decay",
            param_values=[0.999, 0.995, 0.99],
            episodes=30
        )
        
        # -------------------- 3. Action Space Comparison ----------------------
        run_action_space()
        
        # -------------------- 4. Training Length Impact (with longer run) -----
        _ = run_experiment(
            experiment_name="training_length_impact/long_training",
            agent_type="dqn",
            episodes=100,
            steps_per_episode=1000
        )
        
        print("All experiments completed successfully!")
    
    elif args.experiment == 'agent_comparison':
        run_agent_comparison(episodes=args.episodes)
    
    elif args.experiment == 'hyperparam_sweep':
        run_hyperparameter_sweep(
            experiment_name="hyperparameter_sweep/learning_rate",
            agent_type=args.agent,
            param_name="learning_rate",
            param_values=[0.001, 0.0005, 0.0001],
            episodes=args.episodes
        )
        
    elif args.experiment == 'action_space':
        run_action_space(agent_type=args.agent, episodes=args.episodes)
    
    elif args.experiment == 'single_model':
        run_experiment(
            experiment_name="training_length_impact/long_training",
            agent_type=args.agent,
            episodes=args.episodes,
            steps_per_episode=1000
        )