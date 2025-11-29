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
    """Improved Deep Q-Network architecture with layer normalization."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dims: List[int] = [256, 128, 64]):
        super(DQNNetwork, self).__init__()

        layers = []
        input_dim = state_dim

        # Use LayerNorm instead of BatchNorm - works with any batch size
        self.input_norm = nn.LayerNorm(state_dim)

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:  # No dropout on last hidden layer
                layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, n_actions))
        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Handle both batched and unbatched inputs
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Normalize input (LayerNorm works with any batch size)
        x = self.input_norm(x)
        return self.network(x)


class DQNAgent(BaseAgent):
    """
    Deep Q-Network (DQN) agent with experience replay and target network.
    """

    def __init__(self, n_actions: int, state_dim: int,
                 learning_rate: float = 1e-3, epsilon: float = 0.3,
                 epsilon_decay: float = 0.9999, min_epsilon: float = 0.01,
                 gamma: float = 0.99, batch_size: int = 128,
                 memory_size: int = 100000, target_update_freq: int = 100,
                 hidden_dims: List[int] = [256, 128, 64], warmup_steps: int = 1000):
        super().__init__(n_actions, state_dim, "DQN")

        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.warmup_steps = warmup_steps

        # Neural networks
        self.q_network = DQNNetwork(state_dim, n_actions, hidden_dims)
        self.target_network = DQNNetwork(state_dim, n_actions, hidden_dims)
        # Use AdamW with weight decay for better regularization
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=learning_rate, weight_decay=1e-5)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.9)

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network always in eval mode

        # Experience replay buffer with priority (recent experiences more important)
        self.memory = deque(maxlen=memory_size)
        self.recent_memory_size = min(5000, memory_size // 4)  # Keep recent experiences separate
        
        # Track reward variance for adaptive learning
        self.recent_reward_window = deque(maxlen=100)  # Track last 100 rewards
        self.reward_variance = 0.0

        # State normalization
        self.state_mean = np.zeros(state_dim)
        self.state_std = np.ones(state_dim)
        self.state_count = 0

        # Training metrics
        self.losses = []

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using exponential moving average (forgets old data)."""
        # Use exponential moving average instead of full history
        # This allows adaptation to changing preferences
        alpha = 0.01  # Learning rate for normalization
        
        if self.state_count == 0:
            self.state_mean = state.copy()
            self.state_std = np.ones_like(state)
            self.state_count = 1
            return state
        
        # Exponential moving average - forgets old data gradually
        self.state_count += 1
        delta = state - self.state_mean
        self.state_mean += alpha * delta  # EMA update
        
        # Update std with EMA
        delta2 = state - self.state_mean
        variance_ema = (1 - alpha) * (self.state_std ** 2) + alpha * (delta2 ** 2)
        self.state_std = np.sqrt(variance_ema)
        self.state_std = np.clip(self.state_std, 1e-8, None)
        
        return (state - self.state_mean) / self.state_std

    def select_action(self, state: np.ndarray) -> int:
        """Select action using ε-greedy policy with DQN."""
        # Normalize state
        state_normalized = self._normalize_state(state.copy())

        if np.random.random() < self.epsilon or self.step_count < self.warmup_steps:
            # Explore: use softmax for better exploration during warmup
            if self.step_count < self.warmup_steps:
                # More exploration during warmup
                action = np.random.randint(0, self.n_actions)
            else:
                # Random action
                action = np.random.randint(0, self.n_actions)
        else:
            # Exploit: best action according to Q-network
            with torch.no_grad():
                self.q_network.eval()  # Ensure eval mode for inference
                state_tensor = torch.FloatTensor(state_normalized).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()

        self.step_count += 1
        return action

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> None:
        """Update DQN using experience replay."""
        # Normalize states before storing
        state_normalized = self._normalize_state(state.copy())
        next_state_normalized = self._normalize_state(next_state.copy())

        # Store experience in replay buffer
        self.memory.append((state_normalized, action, reward, next_state_normalized, done))
        
        # Update reward variance tracking for adaptive learning
        self.recent_reward_window.append(reward)
        if len(self.recent_reward_window) > 10:
            self.reward_variance = np.var(list(self.recent_reward_window))

        # Train if we have enough experiences (after warmup)
        if len(self.memory) >= self.batch_size and self.step_count >= self.warmup_steps:
            self._train_step()

        # Update target network periodically (soft update for stability)
        if self.step_count % self.target_update_freq == 0 and self.step_count > 0:
            # Soft update: mix old and new weights
            tau = 0.01  # Soft update coefficient
            for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

        # Slower epsilon decay - maintain exploration for adaptation to changes
        if self.epsilon > self.min_epsilon:
            # Slower decay to maintain exploration as preferences change
            # Don't decay too fast - we need exploration to adapt
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # Track metrics
        self.rewards_history.append(reward)
        self.actions_history.append(action)

    def _train_step(self) -> None:
        """Perform one training step using experience replay with recency bias."""
        # Sample batch with bias towards recent experiences
        # This helps adapt to changing preferences
        if len(self.memory) < self.batch_size:
            batch = list(self.memory)
        else:
            # Adaptive sampling: more recent experiences if reward variance is high (preferences changing)
            # High variance indicates preferences are changing, so prioritize recent data more
            if self.reward_variance > 0.1:  # High variance = preferences changing
                recent_ratio = 0.85  # 85% recent, 15% old
            else:
                recent_ratio = 0.70  # 70% recent, 30% old (normal)
            
            recent_size = int(self.batch_size * recent_ratio)
            recent_experiences = list(self.memory)[-self.recent_memory_size:]
            old_experiences = list(self.memory)[:-self.recent_memory_size] if len(self.memory) > self.recent_memory_size else []
            
            if len(recent_experiences) >= recent_size and len(old_experiences) >= (self.batch_size - recent_size):
                recent_batch = random.sample(recent_experiences, recent_size)
                old_batch = random.sample(old_experiences, self.batch_size - recent_size)
                batch = recent_batch + old_batch
            else:
                # Fallback to random sampling if not enough data
                batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)

        # Current Q-values
        self.q_network.train()  # Set to training mode
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Next Q-values from target network (Double DQN: use main network for action selection)
        with torch.no_grad():
            self.target_network.eval()  # Ensure target network is in eval mode
            # Use main network to select best action
            next_actions = self.q_network(next_states).argmax(1)
            # Use target network to evaluate that action
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss with Huber loss (more robust to outliers)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.scheduler.step()

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
            # Normalize state
            state_normalized = self._normalize_state(state.copy())
            state_tensor = torch.FloatTensor(state_normalized).unsqueeze(0)
            self.q_network.eval()  # Set to eval mode
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