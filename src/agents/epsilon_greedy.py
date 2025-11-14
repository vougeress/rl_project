"""
ε-Greedy Multi-Armed Bandit agent.
"""

import numpy as np
from .base_agent import BaseAgent


class EpsilonGreedyBandit(BaseAgent):
    """
    ε-Greedy Multi-Armed Bandit agent.
    Treats each product as an arm, ignores state information.
    """
    
    def __init__(self, n_actions: int, state_dim: int, epsilon: float = 0.3,
                 epsilon_decay: float = 0.9995, min_epsilon: float = 0.05):
        super().__init__(n_actions, state_dim, "EpsilonGreedyBandit")
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Q-values for each action (product)
        self.q_values = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)
        
    def select_action(self, state: np.ndarray) -> int:
        """Select action using ε-greedy policy."""
        if np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.randint(0, self.n_actions)
        else:
            # Exploit: best action
            action = np.argmax(self.q_values)
        
        self.step_count += 1
        return action
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> None:
        """Update Q-values using incremental average."""
        self.action_counts[action] += 1
        
        # Incremental update: Q(a) = Q(a) + α * (R - Q(a))
        # Use adaptive learning rate with minimum threshold
        alpha = max(0.01, 1.0 / self.action_counts[action])  # Minimum learning rate
        self.q_values[action] += alpha * (reward - self.q_values[action])
        
        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        
        # Track metrics
        self.rewards_history.append(reward)
        self.actions_history.append(action)
    
    def get_metrics(self):
        """Get ε-Greedy specific metrics."""
        metrics = super().get_metrics()
        metrics.update({
            'epsilon': self.epsilon,
            'best_action': int(np.argmax(self.q_values)),
            'best_q_value': float(np.max(self.q_values)),
            'avg_q_value': float(np.mean(self.q_values))
        })
        return metrics