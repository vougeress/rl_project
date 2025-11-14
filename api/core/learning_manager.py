"""
Global learning manager for shared agent across all services.
"""

import uuid
import numpy as np
from typing import Dict, Optional
from datetime import datetime

from src.data_generation import generate_synthetic_data
from src.environment import ECommerceEnv
from src.agents.factory import create_agent


class GlobalLearningManager:
    """Singleton manager for shared learning agent."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalLearningManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.catalog = None
            self.simulator = None
            self.env = None
            self.agent = None
            self.is_ready = False
            self.learning_history = []
            GlobalLearningManager._initialized = True
    
    def initialize_system(self, n_products: int = 500, n_users: int = 100) -> Dict:
        """Initialize the learning system."""
        try:
            print("🚀 Initializing global learning system...")
            
            # Generate data
            self.catalog, self.simulator = generate_synthetic_data(n_products, n_users)
            
            # Create environment
            self.env = ECommerceEnv(self.catalog, self.simulator, reward_type='multi_action')
            
            # Create DQN agent
            state_dim = len(self.simulator.get_user_state(0))
            self.agent = create_agent("dqn", n_products, state_dim)
            
            # Pre-train agent
            self._pretrain_agent(episodes=30)
            
            self.is_ready = True
            
            print("✅ Global learning system initialized!")
            return {
                "status": "success",
                "n_products": n_products,
                "n_users": n_users,
                "agent_type": "dqn",
                "pretrain_episodes": 30
            }
            
        except Exception as e:
            print(f"❌ Failed to initialize learning system: {e}")
            return {"status": "error", "error": str(e)}
    
    def _pretrain_agent(self, episodes: int = 30):
        """Pre-train the agent with synthetic episodes."""
        print(f"🧠 Pre-training agent for {episodes} episodes...")
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(min(15, self.env.max_session_length)):
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                # Train the agent
                if hasattr(self.agent, 'update'):
                    self.agent.update(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            self.learning_history.append({
                "episode": episode,
                "reward": episode_reward,
                "type": "pretrain",
                "timestamp": datetime.now()
            })
        
        print(f"✅ Pre-training completed! Average reward: {np.mean([h['reward'] for h in self.learning_history]):.3f}")
    
    def get_recommendations(self, user_id: int, limit: int = 20) -> list:
        """Get recommendations using the trained agent."""
        if not self.is_ready:
            # Auto-initialize if not ready
            self.initialize_system()
        
        # Get user state - ensure user_id is within bounds
        simulator_user_id = user_id % self.simulator.n_users
        user_state = self.simulator.get_user_state(simulator_user_id)
        
        # Generate recommendations
        recommendations = []
        recommended_ids = set()
        
        for _ in range(limit):
            # Get recommendation from agent
            recommended_product_id = self.agent.select_action(user_state)
            
            # Avoid duplicates
            attempts = 0
            while recommended_product_id in recommended_ids and attempts < 50:
                recommended_product_id = self.agent.select_action(user_state)
                attempts += 1
            
            if recommended_product_id not in recommended_ids:
                recommended_ids.add(recommended_product_id)
                product_info = self.catalog.get_product_info(recommended_product_id)
                recommendations.append(product_info)
        
        return recommendations
    
    def learn_from_action(self, user_id: int, product_id: int, action: str, reward: float):
        """Update agent based on user action."""
        if not self.is_ready:
            print("⚠️ Learning system not ready, skipping learning update")
            return
        
        try:
            # Get user state - ensure user_id is within bounds
            simulator_user_id = user_id % self.simulator.n_users
            user_state = self.simulator.get_user_state(simulator_user_id)
            
            # Create next state (simulate progression)
            next_state = self.simulator.get_user_state(simulator_user_id, 1)
            
            # Update agent
            if hasattr(self.agent, 'update'):
                self.agent.update(user_state, product_id, reward, next_state, False)
                
                # Record learning
                self.learning_history.append({
                    "episode": len(self.learning_history),
                    "reward": reward,
                    "type": "user_action",
                    "action": action,
                    "user_id": user_id,
                    "product_id": product_id,
                    "timestamp": datetime.now()
                })
                
                print(f"🧠 Agent learned from '{action}' action (reward: {reward})")
            
        except Exception as e:
            print(f"⚠️ Learning update failed: {e}")
    
    def get_learning_stats(self) -> Dict:
        """Get learning statistics."""
        if not self.is_ready:
            return {"status": "not_ready"}
        
        total_episodes = len(self.learning_history)
        user_actions = [h for h in self.learning_history if h["type"] == "user_action"]
        
        if total_episodes > 0:
            avg_reward = np.mean([h["reward"] for h in self.learning_history])
            recent_rewards = [h["reward"] for h in self.learning_history[-10:]]
            recent_avg = np.mean(recent_rewards) if recent_rewards else 0
        else:
            avg_reward = 0
            recent_avg = 0
        
        # Action distribution
        action_counts = {}
        for h in user_actions:
            action = h.get("action", "unknown")
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            "status": "active",
            "total_episodes": total_episodes,
            "user_interactions": len(user_actions),
            "avg_reward": float(avg_reward),
            "recent_avg_reward": float(recent_avg),
            "action_distribution": action_counts,
            "is_learning": self.is_ready
        }


# Global instance
learning_manager = GlobalLearningManager()