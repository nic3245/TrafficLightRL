import numpy as np
import random
from base_agent import BaseAgent

class LinearQFunction:
    """Linear approximator for Q-function: Q(s,a) = w^T * phi(s,a)"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Initialize weights with separate weight vector for each action
        self.weights = np.random.randn(action_size, state_size) * 0.01
        
        print(f"LinearQFunction created with state_size={state_size}, action_size={action_size}")
        
    def predict(self, state):
        """Compute Q-values for all actions given the state"""
        if not isinstance(state, np.ndarray):
            state = np.array(state)
            
        if np.isnan(state).any():
            print("WARNING: NaN values in state during prediction")
            state = np.nan_to_num(state, nan=0.0)
            
        q_values = np.dot(self.weights, state)
        
        return q_values
    
    def update(self, state, action, target, learning_rate):
        """Update weights using gradient descent"""
        current = np.dot(self.weights[action], state)
        error = target - current
        error = np.clip(error, -10.0, 10.0)
        grad_update = learning_rate * error * state
        grad_update = np.clip(grad_update, -1.0, 1.0)
        self.weights[action] += grad_update
        return min(error ** 2, 100.0)

class LinearAgent(BaseAgent):
    """Agent implementing Q-learning with linear function approximation"""
    def __init__(self, state_size, action_size):
        super(LinearAgent, self).__init__(state_size, action_size)
        self.q_function = LinearQFunction(state_size, action_size)
    
    def act(self, state, training=True):
        """Select an action using the policy"""
        # Regular epsilon-greedy policy
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            norm_state = self.normalize_state(state)
            
            if np.isnan(norm_state).any():
                print(f"WARNING: NaN detected in state during act()!")
                return random.randint(0, self.action_size - 1)
        
            q_values = self.q_function.predict(norm_state)
            
            # Rarely debug print Q-values
            if random.random() < 0.001: 
                print(f"Q-values: {q_values}")
            
            return np.argmax(q_values)
    
    def learn(self):
        """Update value function using a batch of transitions"""
        self.total_learn_calls += 1
        
        # Learn only every 'update_every' steps
        if self.total_learn_calls % self.update_every != 0:
            return None
        
        # Skip if not enough samples
        if len(self.memory) < self.batch_size:
            return None
        
        # Random sample transitions
        batch = random.sample(self.memory, self.batch_size)
        
        total_loss = 0
        
        try:
            # If a weight is NaN, reset it
            if np.isnan(self.q_function.weights).any():
                print("WARNING: NaN detected in weights. Resetting affected weights...")
                nan_mask = np.isnan(self.q_function.weights)
                self.q_function.weights[nan_mask] = np.random.randn(*self.q_function.weights[nan_mask].shape) * 0.01
            
            # Process each transition
            for state, action, reward, next_state, done in batch:
                # Skip any samples with NaN values
                if (np.isnan(state).any() or np.isnan(next_state).any() or 
                    np.isnan(reward) or not np.isfinite(reward)):
                    continue
                
                # Get maximum Q-value for next state
                next_q_values = self.q_function.predict(next_state)
                
                if np.isnan(next_q_values).any() or not np.isfinite(next_q_values).any():
                    # Skip this sample if Q-values have issues
                    continue
                    
                max_next_q = np.max(next_q_values)
                
                # Compute target Q-value
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * max_next_q
                
                # Replace target with reward if target is not finite
                if not np.isfinite(target):
                    target = reward 
                
                # Update weights and compute loss
                loss = self.q_function.update(state, action, target, self.learning_rate)
                
                # Only add loss if it's valid
                if np.isfinite(loss):
                    total_loss += loss
            
            # Average loss over batch
            if self.batch_size > 0:
                avg_loss = total_loss / self.batch_size
            else:
                avg_loss = 0.0
            
            # Check if loss is valid
            if not np.isfinite(avg_loss):
                avg_loss = 0.0
                print("WARNING: Invalid loss detected, setting to 0.0")
            
            self.last_loss = avg_loss
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Print detailed learning info every 1000 learn calls
            if self.total_learn_calls % 1000 == 0:
                print(f"Learning step {self.total_learn_calls}:")
                print(f"  Loss: {avg_loss:.6f}")
                print(f"  Epsilon: {self.epsilon:.4f}")
                
                if not np.isnan(self.q_function.weights).any():
                    weight_mean = np.mean(self.q_function.weights)
                    weight_std = np.std(self.q_function.weights)
                    weight_max = np.max(np.abs(self.q_function.weights))
                    print(f"  Weights stats: mean={weight_mean:.4f}, std={weight_std:.4f}, max_abs={weight_max:.4f}")
                else:
                    print("  WARNING: NaN detected in weights!")
            
            return avg_loss
            
        except Exception as e:
            # Keep going if error
            print(f"Error during learning: {str(e)}")
            return None
    
    def save_model(self, filename):
        """Save the model to a file"""
        np.savez(filename, 
                 weights=self.q_function.weights,
                 state_mean=self.state_mean,
                 state_std=self.state_std,
                 hyperparameters={
                     'gamma': self.gamma,
                     'epsilon': self.epsilon,
                     'epsilon_min': self.epsilon_min,
                     'epsilon_decay': self.epsilon_decay,
                     'learning_rate': self.learning_rate,
                     'batch_size': self.batch_size,
                     'update_every': self.update_every
                 })
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load the model from a file"""
        data = np.load(filename, allow_pickle=True)
        self.q_function.weights = data['weights']
        self.state_mean = data['state_mean']
        self.state_std = data['state_std']
        
        # Load hyperparameters
        hyperparameters = data['hyperparameters'].item()
        self.set_hypers(
            gamma=hyperparameters['gamma'],
            epsilon=hyperparameters['epsilon'],
            epsilon_min=hyperparameters['epsilon_min'],
            epsilon_decay=hyperparameters['epsilon_decay'],
            learning_rate=hyperparameters['learning_rate'],
            batch_size=hyperparameters['batch_size'],
            update_every=hyperparameters['update_every']
        )
        
        print(f"Model loaded from {filename}")