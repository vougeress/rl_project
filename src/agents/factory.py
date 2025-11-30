"""
Factory function for creating RL agents.
"""

from .base_agent import BaseAgent
from .epsilon_greedy import EpsilonGreedyBandit
from .linucb import LinUCBAgent
from .dqn import DQNAgent


def create_agent(agent_type: str, n_actions: int, state_dim: int, **kwargs) -> BaseAgent:
    """
    Factory function to create agents.
    
    Args:
        agent_type: Type of agent to create ('epsilon_greedy', 'linucb', 'dqn')
        n_actions: Number of possible actions
        state_dim: Dimension of state space
        **kwargs: Additional parameters specific to each agent type
    
    Returns:
        BaseAgent: Instance of the requested agent type
    
    Raises:
        ValueError: If agent_type is not recognized
    """
    agent_type = agent_type.lower()
    
    if agent_type in ['epsilon_greedy', 'bandit']:
        return EpsilonGreedyBandit(n_actions, state_dim, **kwargs)
    
    elif agent_type in ['linucb', 'contextual']:
        return LinUCBAgent(n_actions, state_dim, **kwargs)
    
    elif agent_type == 'dqn':
        return DQNAgent(n_actions, state_dim, **kwargs)
    
    else:
        available_types = ['epsilon_greedy', 'linucb', 'dqn']
        raise ValueError(f"Unknown agent type: {agent_type}. Available types: {available_types}")


def get_available_agents():
    """Get list of available agent types."""
    return {
        'epsilon_greedy': {
            'name': 'ε-Greedy Multi-Armed Bandit',
            'description': 'Simple bandit algorithm with epsilon-greedy exploration',
            'parameters': ['epsilon', 'epsilon_decay', 'min_epsilon']
        },
        'linucb': {
            'name': 'Linear Upper Confidence Bound',
            'description': 'Contextual bandit using linear model with confidence bounds',
            'parameters': ['alpha']
        },
        'dqn': {
            'name': 'Deep Q-Network',
            'description': 'Deep reinforcement learning with experience replay',
            'parameters': ['learning_rate', 'epsilon', 'epsilon_decay', 'gamma', 'batch_size', 'memory_size', 'hidden_dims']
        }
    }


def get_default_parameters(agent_type: str):
    """Get default parameters for an agent type."""
    defaults = {
        'epsilon_greedy': {
            'epsilon': 0.1,
            'epsilon_decay': 0.995,
            'min_epsilon': 0.01
        },
        'linucb': {
            'alpha': 1.0
        },
        'dqn': {
            'learning_rate': 1e-3,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'min_epsilon': 0.01,
            'gamma': 0.99,
            'batch_size': 32,
            'memory_size': 10000,
            'target_update_freq': 100,
            'hidden_dims': [128, 64]
        }
    }
    
    return defaults.get(agent_type.lower(), {})