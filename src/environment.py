"""
E-commerce recommendation environment for reinforcement learning.
"""

import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional
from .data_generation import ProductCatalog, UserSimulator


class ECommerceEnv(gym.Env):
    """
    E-commerce recommendation environment.
    
    State: User features + context (session time, previous interactions)
    Action: Product recommendation (product ID)
    Reward: Engagement score (click, purchase probability, dwell time)
    """
    
    def __init__(self, catalog: ProductCatalog, simulator: UserSimulator,
                 max_session_length: int = 50, reward_type: str = 'multi_action'):
        super(ECommerceEnv, self).__init__()
        
        self.catalog = catalog
        self.simulator = simulator
        self.max_session_length = max_session_length
        self.reward_type = reward_type
        
        # Environment state
        self.current_user_id = 0
        self.session_step = 0
        self.session_history = []
        self.cumulative_reward = 0
        
        # Action space: recommend any product
        self.action_space = spaces.Discrete(catalog.n_products)
        
        # Observation space: user state vector
        sample_state = simulator.get_user_state(0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(sample_state),), dtype=np.float32
        )
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_clicks = []
        self.episode_diversity = []
        
    def reset(self, user_id: Optional[int] = None) -> np.ndarray:
        """Reset environment for new episode."""
        if user_id is None:
            self.current_user_id = np.random.randint(0, self.simulator.n_users)
        else:
            self.current_user_id = user_id
            
        self.session_step = 0
        self.session_history = []
        self.cumulative_reward = 0
        
        # Get initial state
        state = self.simulator.get_user_state(self.current_user_id, self.session_step)
        return state.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment."""
        # Validate action
        if action < 0 or action >= self.catalog.n_products:
            raise ValueError(f"Invalid action {action}. Must be in [0, {self.catalog.n_products})")
        
        # Get product features
        product_features = self.catalog.get_product_features(action)
        
        # Simulate user interaction with multiple actions
        interaction_result = self.simulator.simulate_user_interaction(
            self.current_user_id, product_features, self.session_step
        )
        
        # Calculate reward based on occurred actions
        if self.reward_type == 'multi_action':
            # Use total reward from all occurred actions
            reward = interaction_result['total_reward']
        elif self.reward_type == 'engagement':
            # Backward compatibility - use old engagement calculation
            engagement = self.simulator.calculate_engagement(
                self.current_user_id, product_features, self.session_step
            )
            reward = engagement
        elif self.reward_type == 'purchase_focused':
            # Focus on purchase actions
            purchase_actions = [a for a in interaction_result['occurred_actions']
                             if a['action'] == 'purchase']
            reward = sum(a['reward'] for a in purchase_actions) if purchase_actions else 0
        elif self.reward_type == 'engagement_focused':
            # Focus on positive engagement (like, share, add_to_cart)
            positive_actions = [a for a in interaction_result['occurred_actions']
                              if a['action'] in ['like', 'share', 'add_to_cart']]
            reward = sum(a['reward'] for a in positive_actions) if positive_actions else 0
        else:
            # Default to multi-action
            reward = interaction_result['total_reward']
        
        # Update session
        self.session_step += 1
        self.session_history.append({
            'step': self.session_step,
            'action': action,
            'product_features': product_features,
            'reward': reward,
            'occurred_actions': interaction_result['occurred_actions'],
            'action_probabilities': interaction_result['all_action_probs']
        })
        self.cumulative_reward += reward
        
        # Check if episode is done
        done = self.session_step >= self.max_session_length
        
        # Get next state
        next_state = self.simulator.get_user_state(self.current_user_id, self.session_step)
        
        # Calculate detailed metrics
        positive_actions = sum(1 for h in self.session_history
                             for a in h['occurred_actions']
                             if a['reward'] > 0)
        negative_actions = sum(1 for h in self.session_history
                             for a in h['occurred_actions']
                             if a['reward'] < 0)
        purchases = sum(1 for h in self.session_history
                       for a in h['occurred_actions']
                       if a['action'] == 'purchase')
        
        # Additional info
        info = {
            'user_id': self.current_user_id,
            'session_step': self.session_step,
            'cumulative_reward': self.cumulative_reward,
            'product_id': action,
            'product_category': self.catalog.products_df.iloc[action]['category_name'],
            'occurred_actions': interaction_result['occurred_actions'],
            'action_probabilities': interaction_result['all_action_probs'],
            'positive_actions': positive_actions,
            'negative_actions': negative_actions,
            'purchases': purchases
        }
        
        if done:
            # Calculate episode metrics
            episode_reward = self.cumulative_reward
            
            # Count different types of actions
            total_purchases = sum(1 for h in self.session_history
                                for a in h['occurred_actions']
                                if a['action'] == 'purchase')
            total_likes = sum(1 for h in self.session_history
                            for a in h['occurred_actions']
                            if a['action'] == 'like')
            total_cart_adds = sum(1 for h in self.session_history
                                for a in h['occurred_actions']
                                if a['action'] == 'add_to_cart')
            total_negative = sum(1 for h in self.session_history
                               for a in h['occurred_actions']
                               if a['reward'] < 0)
            
            # Calculate diversity (unique categories recommended)
            categories = set()
            for h in self.session_history:
                product_id = h['action']
                category = self.catalog.products_df.iloc[product_id]['category_name']
                categories.add(category)
            episode_diversity = len(categories) / self.catalog.n_categories
            
            # Calculate conversion rates
            purchase_rate = total_purchases / self.session_step if self.session_step > 0 else 0
            engagement_rate = (total_likes + total_cart_adds) / self.session_step if self.session_step > 0 else 0
            negative_rate = total_negative / self.session_step if self.session_step > 0 else 0
            
            self.episode_rewards.append(episode_reward)
            self.episode_clicks.append(total_purchases + total_likes + total_cart_adds)  # Positive interactions
            self.episode_diversity.append(episode_diversity)
            
            info.update({
                'episode_reward': episode_reward,
                'total_purchases': total_purchases,
                'total_likes': total_likes,
                'total_cart_adds': total_cart_adds,
                'total_negative': total_negative,
                'episode_diversity': episode_diversity,
                'purchase_rate': purchase_rate,
                'engagement_rate': engagement_rate,
                'negative_rate': negative_rate
            })
        
        return next_state.astype(np.float32), reward, done, info
    
    def get_valid_actions(self) -> List[int]:
        """Get all valid actions (all products)."""
        return list(range(self.catalog.n_products))
    
    def get_product_info(self, product_id: int) -> Dict:
        """Get detailed product information."""
        if product_id < 0 or product_id >= self.catalog.n_products:
            raise ValueError(f"Invalid product_id {product_id}")
        
        product = self.catalog.products_df.iloc[product_id]
        return {
            'product_id': product_id,
            'category': product['category_name'],
            'price': product['price'],
            'popularity': product['popularity'],
            'quality': product['quality']
        }
    
    def get_user_info(self, user_id: Optional[int] = None) -> Dict:
        """Get user information."""
        if user_id is None:
            user_id = self.current_user_id
            
        if user_id < 0 or user_id >= self.simulator.n_users:
            raise ValueError(f"Invalid user_id {user_id}")
        
        user = self.simulator.users_df.iloc[user_id]
        return {
            'user_id': user_id,
            'age': user['age'],
            'income_level': user['income_level'],
            'price_sensitivity': user['price_sensitivity'],
            'quality_sensitivity': user['quality_sensitivity'],
            'exploration_tendency': user['exploration_tendency']
        }
    
    def get_metrics(self) -> Dict:
        """Get environment metrics."""
        if not self.episode_rewards:
            return {
                'avg_episode_reward': 0,
                'avg_ctr': 0,
                'avg_diversity': 0,
                'total_episodes': 0
            }
        
        return {
            'avg_episode_reward': np.mean(self.episode_rewards),
            'avg_ctr': np.mean([clicks / self.max_session_length for clicks in self.episode_clicks]),
            'avg_diversity': np.mean(self.episode_diversity),
            'total_episodes': len(self.episode_rewards),
            'episode_rewards': self.episode_rewards.copy(),
            'episode_clicks': self.episode_clicks.copy(),
            'episode_diversity': self.episode_diversity.copy()
        }
    
    def render(self, mode: str = 'human') -> None:
        """Render current state."""
        if mode == 'human':
            print(f"=== E-commerce Recommendation Environment ===")
            print(f"Current User: {self.current_user_id}")
            print(f"Session Step: {self.session_step}/{self.max_session_length}")
            print(f"Cumulative Reward: {self.cumulative_reward:.3f}")
            
            if self.session_history:
                last_action = self.session_history[-1]
                product_info = self.get_product_info(last_action['action'])
                print(f"Last Recommendation: {product_info['category']} (${product_info['price']:.2f})")
                print(f"Last Engagement: {last_action['engagement']:.3f}")
            
            user_info = self.get_user_info()
            print(f"User Profile: Age {user_info['age']}, {user_info['income_level']} income")
            print("=" * 50)


class MultiUserECommerceEnv(ECommerceEnv):
    """
    Extended environment that cycles through multiple users.
    Useful for training agents on diverse user behaviors.
    """
    
    def __init__(self, catalog: ProductCatalog, simulator: UserSimulator,
                 max_session_length: int = 50, reward_type: str = 'multi_action',
                 users_per_episode: int = 10):
        super().__init__(catalog, simulator, max_session_length, reward_type)
        self.users_per_episode = users_per_episode
        self.current_user_idx = 0
        self.user_sequence = []
        
    def reset(self, user_ids: Optional[List[int]] = None) -> np.ndarray:
        """Reset with a sequence of users."""
        if user_ids is None:
            # Random sequence of users
            self.user_sequence = np.random.choice(
                self.simulator.n_users, 
                size=self.users_per_episode, 
                replace=True
            ).tolist()
        else:
            self.user_sequence = user_ids
            
        self.current_user_idx = 0
        return super().reset(self.user_sequence[0])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step with automatic user switching."""
        state, reward, done, info = super().step(action)
        
        # If session is done but we have more users, switch to next user
        if done and self.current_user_idx < len(self.user_sequence) - 1:
            self.current_user_idx += 1
            next_user_id = self.user_sequence[self.current_user_idx]
            state = super().reset(next_user_id)
            done = False  # Continue episode with new user
            info['switched_user'] = True
            info['new_user_id'] = next_user_id
        
        return state, reward, done, info


if __name__ == "__main__":
    # Test environment
    from .data_generation import generate_synthetic_data
    
    catalog, simulator = generate_synthetic_data(n_products=100, n_users=10)
    env = ECommerceEnv(catalog, simulator, max_session_length=10)
    
    # Test episode
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    total_reward = 0
    for step in range(10):
        action = np.random.randint(0, env.action_space.n)
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step}: Action={action}, Reward={reward:.3f}, Done={done}")
        if done:
            break
    
    print(f"Total reward: {total_reward:.3f}")
    print(f"Environment metrics: {env.get_metrics()}")