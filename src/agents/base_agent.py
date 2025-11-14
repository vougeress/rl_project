"""
Base agent class for all RL agents.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict


class BaseAgent(ABC):
    """Base class for all RL agents."""
    
    def __init__(self, n_actions: int, state_dim: int, name: str = "BaseAgent"):
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.name = name
        self.step_count = 0
        self.episode_count = 0
        
        # Metrics tracking
        self.rewards_history = []
        self.actions_history = []
        
    @abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        """Select action given current state."""
        pass
    
    @abstractmethod
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> None:
        """Update agent based on experience."""
        pass
    
    def reset_episode(self) -> None:
        """Reset for new episode."""
        self.episode_count += 1
    
    def get_metrics(self) -> Dict:
        """Get agent performance metrics."""
        if not self.rewards_history:
            return {'avg_reward': 0, 'total_steps': 0}
        
        return {
            'avg_reward': np.mean(self.rewards_history[-100:]),  # Last 100 steps
            'total_reward': np.sum(self.rewards_history),
            'total_steps': len(self.rewards_history),
            'episodes': self.episode_count
        }