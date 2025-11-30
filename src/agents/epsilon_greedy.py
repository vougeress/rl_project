"""
ε-Greedy Multi-Armed Bandit agent with contextual improvements.
"""

import numpy as np
from .base_agent import BaseAgent


class EpsilonGreedyBandit(BaseAgent):
    """
    Improved ε-Greedy Multi-Armed Bandit agent.
    Uses state information for better context-aware decisions.
    """

    def __init__(self, n_actions: int, state_dim: int, epsilon: float = 0.2,
                 epsilon_decay: float = 0.9998, min_epsilon: float = 0.01,
                 use_context: bool = True, context_dim: int = 5):
        super().__init__(n_actions, state_dim, "EpsilonGreedyBandit")
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.use_context = use_context

        # Q-values for each action (product)
        self.q_values = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)

        # Contextual Q-values: simple linear model for state-action values
        if use_context:
            # Use reduced context dimension for efficiency
            self.context_dim = min(context_dim, state_dim)
            # Simple linear model: Q(s, a) = w_a^T * s_reduced + b_a
            self.context_weights = np.random.randn(n_actions, self.context_dim) * 0.01
            self.context_biases = np.zeros(n_actions)
            self.context_counts = np.zeros(n_actions)
            self.learning_rate = 0.1

    def select_action(self, state: np.ndarray) -> int:
        """Select action using ε-greedy policy with optional context."""
        if self.use_context and self.step_count > 10:
            # Use contextual Q-values
            state_reduced = state[:self.context_dim] if len(state) >= self.context_dim else state
            contextual_q = (self.context_weights @ state_reduced) + self.context_biases

            # Combine with non-contextual Q-values (weighted average)
            combined_q = 0.7 * contextual_q + 0.3 * self.q_values
        else:
            # Use standard Q-values
            combined_q = self.q_values

        if np.random.random() < self.epsilon:
            # Explore: use softmax for better exploration
            # Temperature decreases over time
            temperature = max(0.1, 1.0 / (self.step_count / 100 + 1))
            exp_q = np.exp((combined_q - np.max(combined_q)) / temperature)
            probs = exp_q / np.sum(exp_q)
            action = np.random.choice(self.n_actions, p=probs)
        else:
            # Exploit: best action (with tie-breaking)
            max_q = np.max(combined_q)
            best_actions = np.where(combined_q == max_q)[0]
            action = np.random.choice(best_actions)

        self.step_count += 1
        return action

    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> None:
        """Update Q-values using incremental average and contextual model with forgetting."""
        self.action_counts[action] += 1
        
        # Update non-contextual Q-values with exponential forgetting
        # Use adaptive learning rate that doesn't decrease too much
        alpha = max(0.1, 1.0 / np.sqrt(self.action_counts[action]))  # Higher minimum learning rate
        self.q_values[action] += alpha * (reward - self.q_values[action])
        
        # Adaptive exponential forgetting - faster when rewards change significantly
        # Calculate if reward is very different from current Q-value (preferences changed)
        reward_error = abs(reward - self.q_values[action])
        
        # Increase forgetting rate if reward error is large (preferences changed)
        if reward_error > 0.5:  # Large error indicates preference change
            forgetting_factor = 0.998  # Forget 0.2% per update (faster adaptation)
        else:
            forgetting_factor = 0.9995  # Normal slow forgetting
        
        # Exponential forgetting for all Q-values (helps adapt to changes)
        self.q_values *= forgetting_factor
        
        # Update contextual Q-values if enabled
        if self.use_context:
            self.context_counts[action] += 1
            state_reduced = state[:self.context_dim] if len(state) >= self.context_dim else state
            
            # Predict current Q-value
            predicted_q = (self.context_weights[action] @ state_reduced) + self.context_biases[action]
            
            # Update with gradient descent - higher learning rate for adaptation
            error = reward - predicted_q
            adaptive_lr = max(0.05, self.learning_rate / np.sqrt(self.context_counts[action]))
            
            # Update weights and bias
            self.context_weights[action] += adaptive_lr * error * state_reduced
            self.context_biases[action] += adaptive_lr * error
        
        # Slower epsilon decay - maintain exploration for adaptation
        if self.epsilon > self.min_epsilon:
            # Slower decay to maintain exploration as preferences change
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
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