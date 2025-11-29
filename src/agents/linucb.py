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
    
    def __init__(self, n_actions: int, state_dim: int, alpha: float = 1.0):
        super().__init__(n_actions, state_dim, "LinUCB")
        self.alpha = alpha  # Confidence parameter (reduced from 2.0 for better exploitation)
        
        # Linear model parameters for each action with better initialization
        # Increased regularization for better stability
        self.lambda_reg = 1.0  # L2 regularization parameter
        self.A = np.array([np.eye(state_dim) * self.lambda_reg for _ in range(n_actions)])  # (n_actions, state_dim, state_dim)
        self.b = np.zeros((n_actions, state_dim))  # (n_actions, state_dim)
        
        # State normalization tracking
        self.state_mean = np.zeros(state_dim)
        self.state_std = np.ones(state_dim)
        self.state_count = 0
        
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using exponential moving average (forgets old data)."""
        # Use exponential moving average instead of full history
        # This allows adaptation to changing preferences
        alpha = 0.01  # Learning rate for normalization (small = more stable, large = more adaptive)
        
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
        self.state_std = np.clip(self.state_std, 1e-8, None)  # Avoid division by zero
        
        # Normalize
        normalized = (state - self.state_mean) / self.state_std
        return normalized
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using LinUCB algorithm."""
        # Normalize state for better numerical stability
        state_normalized = self._normalize_state(state.copy())
        state = state_normalized.reshape(-1, 1)  # Column vector
        ucb_values = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            try:
                # Use Cholesky decomposition for numerical stability
                A_reg = self.A[a] + np.eye(self.state_dim) * 1e-6
                L = np.linalg.cholesky(A_reg)
                L_inv = np.linalg.inv(L)
                A_inv = L_inv.T @ L_inv
                
                # Compute theta (parameter estimate)
                theta = A_inv @ self.b[a]
                
                # Compute confidence bound with adaptive alpha
                # Reduce confidence as we get more data (logarithmic decay)
                adaptive_alpha = self.alpha * np.sqrt(np.log(max(2, self.step_count + 1)) / max(1, self.step_count + 1))
                confidence = adaptive_alpha * np.sqrt(state.T @ A_inv @ state)[0, 0]
                
                # UCB value = predicted reward + confidence bound
                predicted_reward = theta.T @ state.flatten()
                ucb_values[a] = predicted_reward + confidence
            except (np.linalg.LinAlgError, ValueError):
                # Fallback: use optimistic initialization
                ucb_values[a] = 1.0 + np.random.random() * 0.1
        
        action = np.argmax(ucb_values)
        self.step_count += 1
        return action
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> None:
        """Update linear model parameters with exponential forgetting."""
        # Normalize state before update
        state_normalized = self._normalize_state(state.copy())
        state = state_normalized.reshape(-1, 1)  # Column vector
        
        # Adaptive exponential forgetting - faster forgetting when rewards change significantly
        # This helps adapt to changing preferences more quickly
        base_forgetting = 0.999  # Base forgetting rate
        
        # Adaptive forgetting: if reward is very different from prediction, forget faster
        try:
            A_inv = np.linalg.inv(self.A[action] + np.eye(self.state_dim) * 1e-6)
            predicted_reward = (A_inv @ self.b[action]).T @ state.flatten()
            reward_error = abs(reward - predicted_reward)
            
            # Increase forgetting rate if prediction error is large (preferences changed)
            if reward_error > 0.5:  # Large error indicates preference change
                forgetting_factor = 0.995  # Forget 0.5% per update (5x faster)
            else:
                forgetting_factor = base_forgetting
        except:
            forgetting_factor = base_forgetting
        
        # Update A and b for the selected action with adaptive forgetting
        self.A[action] = forgetting_factor * self.A[action] + state @ state.T
        self.b[action] = forgetting_factor * self.b[action] + reward * state.flatten()
        
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