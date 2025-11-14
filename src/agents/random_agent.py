"""
Random baseline agent for comparison.
"""

import numpy as np
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Random baseline agent for comparison."""
    
    def __init__(self, n_actions: int, state_dim: int):
        super().__init__(n_actions, state_dim, "Random")
    
    def select_action(self, state: np.ndarray) -> int:
        """Select random action."""
        return np.random.randint(0, self.n_actions)
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> None:
        """No learning for random agent."""
        self.rewards_history.append(reward)
        self.actions_history.append(action)
    
    def get_metrics(self):
        """Get Random agent specific metrics."""
        metrics = super().get_metrics()
        metrics.update({
            'strategy': 'random',
            'action_distribution': self._get_action_distribution()
        })
        return metrics
    
    def _get_action_distribution(self):
        """Get distribution of actions taken."""
        if not self.actions_history:
            return {}
        
        action_counts = {}
        for action in self.actions_history:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Convert to percentages
        total_actions = len(self.actions_history)
        action_distribution = {
            action: count / total_actions 
            for action, count in action_counts.items()
        }
        
        return action_distribution