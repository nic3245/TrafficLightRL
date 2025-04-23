# DQN agent implementation (dqn_agent.py)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from base_agent import BaseAgent

class QNetwork(nn.Module):
    """Standard Q-Network architecture"""
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, action_size)
        
        # Initialize each layer
        self.apply(self._init_weights)
        
        print(f"QNetwork created with state_size={state_size}, action_size={action_size}")
    
    def _init_weights(self, module):
        """Initialize weights in a Linear Layer using Xavier initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent(BaseAgent):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__(state_size, action_size)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.qnetwork_local = QNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(self.device)
        # Copy initial weights to target network
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber loss for stability
        
        # Additional hyperparameter for DQN
        self.tau = 0.001
        
    def set_hypers(self, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999, learning_rate=0.0005, 
                   batch_size=64, update_every=4, tau=0.001, min_phase_duration=10):
        """Override to add tau hyperparameter"""
        super().set_hypers(gamma, epsilon, epsilon_min, epsilon_decay, learning_rate, 
                          batch_size, update_every, min_phase_duration)
        self.tau = tau
        
        # Update optimizer with new learning rate
        if hasattr(self, 'optimizer'): # Make sure it exists first
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
        
    def normalize_state(self, state):
        """Override to ensure torch tensors"""
        normalized = super().normalize_state(state)
        
        # Convert to tensor if not already
        if not isinstance(normalized, torch.Tensor):
            normalized = torch.tensor(normalized, dtype=torch.float32)
            
        return normalized
        
    def act(self, state, training=True):
        """Select an action using the policy"""
        # Regular epsilon-greedy policy
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            norm_state = self.normalize_state(state)
            
            if torch.isnan(norm_state).any():
                print(f"WARNING: NaN detected in state during act()!")
                return random.randint(0, self.action_size - 1)
                
            norm_state = norm_state.to(self.device)
            self.qnetwork_local.eval() # switch to eval mode for speed
            with torch.no_grad():
                q_values = self.qnetwork_local(norm_state)
                
                # Rarely debug print Q-values
                if random.random() < 0.001:
                    print(f"Q-values: {q_values.cpu().numpy()}")
                    
            self.qnetwork_local.train() # switch back to train mode
            return torch.argmax(q_values).item()
           
    def learn(self):
        self.total_learn_calls += 1
        
        # Only learn every 'update_every' steps
        if self.total_learn_calls % self.update_every != 0:
            return None
        
        # Skip if not enough samples
        if len(self.memory) < self.batch_size:
            return None
            
        # Random sample transitions
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        try:
            # Convert the batch to tensors and move to device for speed
            states = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32) for s in states]).to(self.device)
            actions = torch.tensor(actions, dtype=torch.long).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            next_states = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32) for s in next_states]).to(self.device)
            dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
            
            # Ensure model is in training mode
            self.qnetwork_local.train()
            
            # Get 'ideal' Q values from target network + Bellman equation
            with torch.no_grad():
                next_q_values = self.qnetwork_target(next_states).max(1)[0]
                next_q_values[dones] = 0.0
                
                target_q_values = rewards + self.gamma * next_q_values
            
            # Get expected Q values from local model
            expected_q_values = self.qnetwork_local(states).gather(1, actions.unsqueeze(1))
            
            # Compute Huber loss between expected and target Q values
            loss = self.criterion(expected_q_values, target_q_values.unsqueeze(1))
            
            # Print detailed learning info every 1000 learn calls
            if self.total_learn_calls % 1000 == 0:
                print(f"---- Learning step {self.total_learn_calls}: ----")
                print(f"  Expected Q mean: {expected_q_values.mean().item():.4f}")
                print(f"  Target Q mean: {target_q_values.mean().item():.4f}")
                print(f"  Loss: {loss.item():.6f}")
                print(f"  Epsilon: {self.epsilon:.4f}")
            
            # Backpropagate
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
            
            self.optimizer.step()
            
            self.last_loss = loss.item()
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            return loss.item()
            
        except Exception as e: 
            # Keep going if error
            print(f"Error during learning: {str(e)}")
            return None
            
    def update_target_network(self):
        """Soft update of target network"""
        # Using soft update with tau parameter
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
            
        # Print update every 1000 learn calls
        if self.total_learn_calls % 1000 == 0:
            print("Target network updated")
            
    def save_model(self, filename):
        """Save the model to a file"""
        torch.save({
            'qnetwork_local_state_dict': self.qnetwork_local.state_dict(),
            'qnetwork_target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'hyperparameters': {
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'update_every': self.update_every,
                'tau': self.tau
            }
        }, filename)
        print(f"Model saved to {filename}")
        
    def load_model(self, filename):
        """Load the model from a file"""
        checkpoint = torch.load(filename)
        self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local_state_dict'])
        self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.state_mean = checkpoint['state_mean']
        self.state_std = checkpoint['state_std']
        
        # Load hyperparameters
        hyperparameters = checkpoint['hyperparameters']
        self.set_hypers(
            gamma=hyperparameters['gamma'],
            epsilon=hyperparameters['epsilon'],
            epsilon_min=hyperparameters['epsilon_min'],
            epsilon_decay=hyperparameters['epsilon_decay'],
            learning_rate=hyperparameters['learning_rate'],
            batch_size=hyperparameters['batch_size'],
            update_every=hyperparameters['update_every'],
            tau=hyperparameters['tau']
        )
        
        print(f"Model loaded from {filename}")