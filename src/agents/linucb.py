"""
Linear Upper Confidence Bound (LinUCB) Contextual Bandit agent.
"""

import numpy as np
from .base_agent import BaseAgent


class LinUCBAgent(BaseAgent):
    """
    Linear Upper Confidence Bound (LinUCB) Contextual Bandit agent.
    Uses linear model to predict rewards based on context (state).
    """
    
    def __init__(self, n_actions: int, state_dim: int, alpha: float = 2.0):
        super().__init__(n_actions, state_dim, "LinUCB")
        self.alpha = alpha  # Confidence parameter
        
        # Linear model parameters for each action with regularization
        self.A = np.array([np.eye(state_dim) * 0.1 for _ in range(n_actions)])  # (n_actions, state_dim, state_dim)
        self.b = np.zeros((n_actions, state_dim))  # (n_actions, state_dim)
        
    def select_action(self, state: np.ndarray) -> int:
        """Select action using LinUCB algorithm."""
        state = state.reshape(-1, 1)  # Column vector
        ucb_values = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            try:
                # Compute theta (parameter estimate) with regularization
                A_reg = self.A[a] + np.eye(self.state_dim) * 1e-6  # Add small regularization
                A_inv = np.linalg.inv(A_reg)
                theta = A_inv @ self.b[a]
                
                # Compute confidence bound
                confidence = self.alpha * np.sqrt(state.T @ A_inv @ state)[0, 0]
                
                # UCB value = predicted reward + confidence bound
                ucb_values[a] = theta.T @ state.flatten() + confidence
            except np.linalg.LinAlgError:
                # Fallback for singular matrices
                ucb_values[a] = np.random.random()
        
        action = np.argmax(ucb_values)
        self.step_count += 1
        return action
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> None:
        """Update linear model parameters."""
        state = state.reshape(-1, 1)  # Column vector
        
        # Update A and b for the selected action
        self.A[action] += state @ state.T
        self.b[action] += reward * state.flatten()
        
        # Track metrics
        self.rewards_history.append(reward)
        self.actions_history.append(action)
    
    def get_metrics(self):
        """Get LinUCB specific metrics."""
        metrics = super().get_metrics()
        
        # Calculate average confidence bounds
        try:
            avg_confidence = 0
            for a in range(self.n_actions):
                A_inv = np.linalg.inv(self.A[a])
                # Use identity state for average confidence calculation
                identity_state = np.ones((self.state_dim, 1)) / np.sqrt(self.state_dim)
                confidence = self.alpha * np.sqrt(identity_state.T @ A_inv @ identity_state)[0, 0]
                avg_confidence += confidence
            avg_confidence /= self.n_actions
            
            metrics.update({
                'alpha': self.alpha,
                'avg_confidence': float(avg_confidence),
                'model_updates': int(np.sum([np.trace(A) for A in self.A]) - self.n_actions * self.state_dim)
            })
        except np.linalg.LinAlgError:
            # Handle singular matrix case
            metrics.update({
                'alpha': self.alpha,
                'avg_confidence': 0.0,
                'model_updates': 0
            })
        
        return metrics
    
    def get_action_predictions(self, state: np.ndarray) -> np.ndarray:
        """Get predicted rewards for all actions given a state."""
        state = state.reshape(-1, 1)
        predictions = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            try:
                A_inv = np.linalg.inv(self.A[a])
                theta = A_inv @ self.b[a]
                predictions[a] = theta.T @ state.flatten()
            except np.linalg.LinAlgError:
                predictions[a] = 0.0
        
        return predictions
    
    def get_action_confidences(self, state: np.ndarray) -> np.ndarray:
        """Get confidence bounds for all actions given a state."""
        state = state.reshape(-1, 1)
        confidences = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            try:
                A_inv = np.linalg.inv(self.A[a])
                confidences[a] = self.alpha * np.sqrt(state.T @ A_inv @ state)[0, 0]
            except np.linalg.LinAlgError:
                confidences[a] = float('inf')  # High confidence when matrix is singular
        
        return confidences