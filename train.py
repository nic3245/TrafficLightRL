from simulation import Simulator
from dqn_model import DQNAgent
from linear_approximator import LinearAgent
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import os

def get_agent(agent_type, state_size, action_size):
    """Create an agent of the specified type"""
    if agent_type.lower() == 'dqn':
        return DQNAgent(state_size, action_size)
    elif agent_type.lower() == 'linear':
        return LinearAgent(state_size, action_size)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. Choose 'dqn' or 'linear'.")

def get_save_extension(agent_type):
    """Get the appropriate file extension for saving models"""
    if agent_type.lower() == 'dqn':
        return '.pth'
    elif agent_type.lower() == 'linear':
        return '.npz'
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def plot_training_results(rewards, epsilons, losses, q_values, waiting_vehicles, phase_changes, save_path=None):
    """Plot training metrics and save to file if requested"""
    
    # Create figure with 6 subplots
    _, axes = plt.subplots(3, 2, figsize=(15, 15))
    
    # Plot rewards
    ax1 = axes[0, 0]
    ax1.plot(rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    
    # Add moving average for rewards
    window_size = min(10, len(rewards))
    if window_size > 0:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(rewards)), moving_avg, 'r--', label=f'{window_size}-episode Moving Avg')
        ax1.legend()
    
    # Plot epsilon decay
    ax2 = axes[0, 1]
    ax2.plot(epsilons)
    ax2.set_title('Epsilon Decay')
    ax2.set_xlabel('Episode') 
    ax2.set_ylabel('Epsilon')
    
    # Plot loss
    ax3 = axes[1, 0]
    ax3.plot(losses)
    ax3.set_title('Training Loss')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average Loss')
    
    # Plot average Q-values
    ax4 = axes[1, 1]
    ax4.plot(q_values)
    ax4.set_title('Average Q-Values')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Average Q-Value')
    
    # Plot waiting vehicles
    ax5 = axes[2, 0]
    ax5.plot(waiting_vehicles)
    ax5.set_title('Average Waiting Vehicles')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Avg # of Waiting Vehicles')
    
    # Plot phase changes
    ax6 = axes[2, 1]
    ax6.plot(phase_changes)
    ax6.set_title('Traffic Light Phase Changes')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Number of Phase Changes')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training plot saved to {save_path}")
    
    plt.close()

def train_model(agent_type='dqn', episodes=100, steps_per_episode=1000, debug=True, 
                load_checkpoint=None, save_checkpoint=True, action_space_size=5,
                hyperparams=None):
    """
    Train a reinforcement learning agent
    
    Args:
        agent_type: Type of agent ('dqn' or 'linear')
        episodes: Number of episodes to train
        steps_per_episode: Number of steps per episode
        debug: Whether to print debug information
        load_checkpoint: Path to load checkpoint from
        save_checkpoint: Whether to save checkpoints
        action_space_size: Size of action space (4 or 5)
        hyperparams: Dictionary of hyperparameters to override defaults
    
    Returns:
        Tuple of training metrics
    """
    # Initialize tracking variables
    rewards_history = []
    epsilon_history = []
    loss_history = []
    avg_q_values_history = []
    waiting_vehicles_history = []
    phase_changes_history = []
    
    # Directory and extension for saving models
    save_dir = f"models_{agent_type.lower()}_actions{action_space_size}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_ext = get_save_extension(agent_type)
    
    # Create simulator and agent
    sim = Simulator("intersection_1_1", action_space_size=action_space_size)
    state_size = len(state)
    print(f"State size: {state_size}, Action size: {action_space_size}")
    agent = get_agent(agent_type, state_size, action_space_size)
    
    # Apply hyperparams if provided
    if hyperparams:
        agent.set_hypers(**hyperparams)
        print(f"Applied custom hyperparameters: {hyperparams}")
    
    # Load checkpoint if provided
    if load_checkpoint:
        if os.path.exists(load_checkpoint):
            agent.load_model(load_checkpoint)
            print(f"Loaded model from {load_checkpoint}")
        else:
            print(f"Checkpoint {load_checkpoint} not found, starting fresh")
    
    # Start training
    start_time = time.time()
    best_reward = float('-inf')

    for episode in range(episodes):
        # Reset simulator
        sim.reset()
        state, _ = sim.step_simulation(0, step=1)
        
        # Track metrics for this episode
        episode_start = time.time()
        total_reward = 0
        episode_losses = []
        episode_q_values = []
        actions_taken = np.zeros(action_space_size)
        phase_changes = 0
        last_action = 0
        waiting_vehicles_counts = []
        
        for step in range(steps_per_episode):
            
            # Select and perform action
            action = agent.act(state, training=True)
            actions_taken[action] += 1
            
            # Track phase changes 
            if action != last_action:
                phase_changes += 1
                last_action = action
            
            # Execute action
            next_state, reward = sim.step_simulation(action, step=1)
            
            # Track waiting vehicles
            waiting_vehicles_count = len(sim.waiting_vehicles.keys())
            waiting_vehicles_counts.append(waiting_vehicles_count)
            
            # Store transition and learn
            agent.store_transition(state, action, reward, next_state, False)
            loss = agent.learn()
            
            if loss is not None:
                episode_losses.append(loss)
            
            # Update target network for DQN
            if agent_type.lower() == 'dqn':
                agent.update_target_network()
            
            # Move to next state
            state = next_state
            total_reward += reward
            
            # Debug what's happening
            if debug and step % 500 == 0:
                print(f" -------- Step {step}: -------- ")  
                print(f"  Step {step}: Taking action {action}, epsilon: {agent.epsilon:.4f}")
                print(f"  Waiting vehicles: {waiting_vehicles_count}")
                print(f"  Reward: {reward:.2f}, Total so far: {total_reward:.2f}")
                
                # Get Q-values for monitoring
                norm_state = agent.normalize_state(state)
                
                if agent_type.lower() == 'dqn':
                    with torch.no_grad():
                        q_values = agent.qnetwork_local(norm_state).cpu().numpy()
                else:  # Linear
                    q_values = agent.q_function.predict(norm_state)
                
                avg_q = np.mean(q_values)
                min_q = np.min(q_values)
                max_q = np.max(q_values)
                print(f"  Q-values: avg={avg_q:.2f}, min={min_q:.2f}, max={max_q:.2f}")
                episode_q_values.append(avg_q)
                
                if episode_losses:
                    avg_loss = np.mean(episode_losses[-100:])
                    print(f"  Memory size: {len(agent.memory)}/{agent.memory.maxlen}, Avg loss: {avg_loss:.6f}")
                
        # End of episode stats
        episode_time = time.time() - episode_start
        rewards_history.append(total_reward)
        epsilon_history.append(agent.epsilon)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        loss_history.append(avg_loss)
        avg_q_value = np.mean(episode_q_values) if episode_q_values else 0
        avg_q_values_history.append(avg_q_value)
        avg_waiting_vehicles = np.mean(waiting_vehicles_counts)
        waiting_vehicles_history.append(avg_waiting_vehicles)
        phase_changes_history.append(phase_changes)
        
        # Print episode summary
        print(f"---------------------------------- Episode {episode+1} ----------------------------------")
        print(f"Episode: {episode+1}/{episodes} - Time: {episode_time:.1f}s")
        print(f"  Total reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}, Avg Loss: {avg_loss:.6f}")
        print(f"  Avg Q-value: {avg_q_value:.2f}, Avg waiting vehicles: {avg_waiting_vehicles:.1f}")
        print(f"  Phase changes: {phase_changes}, Actions distribution: {actions_taken}")
        
        # Print rolling averages
        if episode > 0:
            window = min(10, episode)
            recent_rewards = rewards_history[-window:]
            print(f"  Avg reward (last {window} episodes): {np.mean(recent_rewards):.2f}")
            
        # Save best model
        if save_checkpoint and (total_reward > best_reward or episode % 10 == 0):
            if total_reward > best_reward:
                best_reward = total_reward
                agent.save_model(f"{save_dir}/best_model{save_ext}")
            
            # Also save periodic checkpoints
            if episode % 10 == 0:
                agent.save_model(f"{save_dir}/checkpoint_episode_{episode}{save_ext}")
                
            # Save training curves
            plot_training_results(
                rewards_history, epsilon_history, loss_history, 
                avg_q_values_history, waiting_vehicles_history, phase_changes_history,
                save_path=f"{save_dir}/training_curves_episode_{episode}.png"
            )
        
        print("-" * 70)
    
    # Training complete
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f} seconds")
    print(f"Final average reward (last 10 episodes): {np.mean(rewards_history[-10:]):.2f}")
    
    # Save final model
    if save_checkpoint:
        agent.save_model(f"{save_dir}/final_model{save_ext}")
    
    return rewards_history, epsilon_history, loss_history, avg_q_values_history, waiting_vehicles_history, phase_changes_history
