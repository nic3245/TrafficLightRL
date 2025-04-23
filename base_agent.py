from abc import ABC, abstractmethod
import numpy as np
from collections import deque
import torch

class BaseAgent(ABC):
    """Abstract base class for reinforcement learning agents"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.set_hypers()
        
        # Replay buffer
        self.memory = deque(maxlen=10000)
        
        # Track stats for debugging
        self.last_loss = None
        self.total_learn_calls = 0
        self.total_transitions = 0
        
        # Track rewards for normalization
        self.reward_history = deque(maxlen=1000)
        
        # Track states for normalization
        self.state_mean = None
        self.state_std = None
        self.num_states_seen = 0
        
    def set_hypers(self, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999, learning_rate=0.01, 
                   batch_size=64, update_every=4, min_phase_duration=10, verbose=False):
        """Set hyperparameters for the agent"""
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_every = update_every
        self.min_phase_duration = min_phase_duration

        if verbose:
            print(f"Agent hyperparameters:")
            print(f"  gamma={gamma}, epsilon={epsilon}, epsilon_min={epsilon_min}")
            print(f"  epsilon_decay={epsilon_decay}, learning_rate={learning_rate}, batch_size={batch_size}")
            print(f"  update_every={update_every}")
            print(f"  min_phase_duration={min_phase_duration}")
    
    def update_state_stats(self, state):
        """Update running statistics for state normalization"""
        if isinstance(state, np.ndarray):
            state_array = state
        else:
            if isinstance(state, torch.Tensor):
                state_array = state.cpu().numpy()
            else:
                state_array = np.array(state)
                
        if self.state_mean is None:
            self.state_mean = state_array.copy()
            self.state_std = np.ones_like(state_array)
            self.num_states_seen = 1
        else: # Incremental update of mean and standard deviation
            # Update mean
            self.num_states_seen += 1
            delta = state_array - self.state_mean
            self.state_mean += delta / self.num_states_seen
            # Update std dev (Welford's online algorithm) - https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
            delta2 = state_array - self.state_mean
            M2 = self.state_std * self.state_std * (self.num_states_seen - 1) + delta * delta2
            self.state_std = np.sqrt(M2 / self.num_states_seen + 1e-6)
    
    def normalize_state(self, state):
        """Normalize state using running statistics"""
        is_tensor = False
        if not isinstance(state, np.ndarray):
            if isinstance(state, torch.Tensor):
                is_tensor = True
                state_array = state.cpu().numpy()
            else:
                state_array = np.array(state)
        else:
            state_array = state
            
        if self.state_mean is None:
            normalized = state_array
        else:
            normalized = (state_array - self.state_mean) / (self.state_std + 1e-6)
        
        if is_tensor:
            return torch.tensor(normalized, dtype=torch.float32)
        else:
            return normalized
    
    def normalize_reward(self, reward):
        """Normalize rewards to prevent numerical instability"""
        # Hard clipping to handle extremely large values
        reward = np.clip(reward, -10000, 10000)
        
        # Track rewards
        self.reward_history.append(reward)
        
        # If we don't have enough data yet, basic clipping
        if len(self.reward_history) < 10:
            return np.clip(reward, -10, 10)
        
        rewards_array = np.array(self.reward_history)
        
        # Normalize based on running statistics with wider std dev
        reward_mean = np.mean(rewards_array)
        reward_std = np.std(rewards_array) + 1e-4
        
        # Clip normalized reward to prevent outliers
        normalized_reward = np.clip((reward - reward_mean) / reward_std, -3, 3)
        
        return normalized_reward
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""

        self.update_state_stats(state)
        self.update_state_stats(next_state)
        
        norm_state = self.normalize_state(state)
        norm_next_state = self.normalize_state(next_state)
        
        normalized_reward = self.normalize_reward(reward)
        
        if np.isnan(np.array(norm_state)).any() or np.isnan(np.array(norm_next_state)).any():
            print(f"WARNING: NaN detected in state or next_state!")
            return
        
        self.memory.append((norm_state, action, normalized_reward, norm_next_state, done))
        self.total_transitions += 1
    
    @abstractmethod
    def act(self, state, training=True):
        """Select an action using policy"""
        pass
    
    @abstractmethod
    def learn(self):
        """Update value function using a batch of transitions"""
        pass
    
    @abstractmethod
    def save_model(self, filename):
        """Save the model to a file"""
        pass
    
    @abstractmethod
    def load_model(self, filename):
        """Load the model from a file"""
        pass