"""
Deep Q-Network (DQN) agent with experience replay and target network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from typing import List
import random

from .base_agent import BaseAgent


class DQNNetwork(nn.Module):
    """Deep Q-Network architecture."""
    
    def __init__(self, state_dim: int, n_actions: int, hidden_dims: List[int] = [128, 64]):
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, n_actions))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class DQNAgent(BaseAgent):
    """
    Deep Q-Network (DQN) agent with experience replay and target network.
    """
    
    def __init__(self, n_actions: int, state_dim: int,
                 learning_rate: float = 3e-4, epsilon: float = 1.0,
                 epsilon_decay: float = 0.9995, min_epsilon: float = 0.05,
                 gamma: float = 0.95, batch_size: int = 64,
                 memory_size: int = 50000, target_update_freq: int = 200,
                 hidden_dims: List[int] = [256, 128, 64]):
        super().__init__(n_actions, state_dim, "DQN")
        
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Neural networks
        self.q_network = DQNNetwork(state_dim, n_actions, hidden_dims)
        self.target_network = DQNNetwork(state_dim, n_actions, hidden_dims)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Training metrics
        self.losses = []
        
    def select_action(self, state: np.ndarray) -> int:
        """Select action using ε-greedy policy with DQN."""
        if np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.randint(0, self.n_actions)
        else:
            # Exploit: best action according to Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
        
        self.step_count += 1
        return action
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> None:
        """Update DQN using experience replay."""
        # Store experience in replay buffer
        self.memory.append((state, action, reward, next_state, done))
        
        # Train if we have enough experiences
        if len(self.memory) >= self.batch_size:
            self._train_step()
        
        # Update target network periodically
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        
        # Track metrics
        self.rewards_history.append(reward)
        self.actions_history.append(action)
    
    def _train_step(self) -> None:
        """Perform one training step using experience replay."""
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.losses.append(loss.item())
    
    def get_metrics(self):
        """Get DQN-specific metrics."""
        metrics = super().get_metrics()
        metrics.update({
            'epsilon': self.epsilon,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'memory_size': len(self.memory),
            'total_updates': len(self.losses)
        })
        return metrics
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions given a state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.squeeze().numpy()
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']