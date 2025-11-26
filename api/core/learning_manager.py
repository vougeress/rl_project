"""
Global learning manager for shared agent across all services.
"""

import numpy as np
import asyncio
from datetime import datetime
from collections import defaultdict
from datetime import datetime, timedelta
from sqlalchemy import select
from typing import Dict, List

from api.models.database_models import Product, User, UserAction
from src.data_generation import generate_synthetic_data
from src.environment import ECommerceEnv
from src.agents.factory import create_agent
from api.core.database import AsyncSessionLocal


class GlobalLearningManager:
    """Singleton manager for shared learning agent."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not getattr(self, '_initialized', False):
            self.catalog = None
            self.simulator = None
            self.env = None
            self.agent = None
            self.is_ready = False
            self.learning_history = []
            self._initialization_lock = asyncio.Lock()
            self._initialized = True
    
    async def initialize_system(self, n_products: int = 500, n_users: int = 100):
        """Initialize the learning system asynchronously."""
        async with self._initialization_lock:
            if self.is_ready:
                print("✅ Learning system already initialized")
                return True
            
            try:
                print("🚀 Initializing global learning system...")
                
                # Generate data
                self.catalog, self.simulator = generate_synthetic_data(n_products, n_users)

                await self._save_products_to_db()
                
                # Create environment
                self.env = ECommerceEnv(self.catalog, self.simulator, reward_type='multi_action')
                
                # Create DQN agent
                state_dim = len(self.simulator.get_user_state(0))
                self.agent = create_agent("dqn", n_products, state_dim)
                
                # Pre-train agent asynchronously
                await self._pretrain_agent(episodes=30)
                
                self.is_ready = True
                print("✅ Global learning system initialized!")
                return True
                
            except Exception as e:
                print(f"❌ Failed to initialize learning system: {e}")
                self.is_ready = False
                return False
    
    async def _save_products_to_db(self):
        """Save synthetic products to database"""
        try:
            async with AsyncSessionLocal() as session:
                from api.models.database_models import UserAction
                await session.execute(UserAction.__table__.delete())
                await session.execute(Product.__table__.delete())
                
                for product_id, row in self.catalog.products_df.iterrows():
                    category_mapping = {
                        'Electronics': 0, 'Clothing': 1, 'Books': 2, 
                        'Home & Garden': 3, 'Sports': 4, 'Beauty': 5,
                        'Toys': 6, 'Automotive': 7, 'Health': 8, 'Food': 9
                    }
                    
                    product = Product(
                        product_id=int(product_id),
                        product_name=row['product_name'],
                        name_format=row.get('name_format'),
                        category_id=category_mapping.get(row['category_name'], 0),
                        category_name=row['category_name'],
                        price=float(row['price']),
                        popularity=float(row['popularity']),
                        quality=float(row['quality']),
                        style_vector={f'style_{i}': float(row[f'style_{i}']) for i in range(5)}
                    )
                    session.add(product)
                
                await session.commit()
                print(f"✅ Saved {len(self.catalog.products_df)} products to database")
                
        except Exception as e:
            print(f"❌ Failed to save products to DB: {e}")
    
    async def _pretrain_agent(self, episodes: int = 30):
        """Pre-train the agent asynchronously."""
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
                
                # Yield control to event loop periodically
                if step % 5 == 0:
                    await asyncio.sleep(0)
            
            self.learning_history.append({
                "episode": episode,
                "reward": episode_reward,
                "type": "pretrain",
                "timestamp": datetime.now()
            })
            
            # Yield control between episodes
            await asyncio.sleep(0)
        
        avg_reward = np.mean([h['reward'] for h in self.learning_history])
        print(f"✅ Pre-training completed! Average reward: {avg_reward:.3f}")
    
    async def get_recommendations(self, user_id: int, limit: int = 20) -> list:
        """Get recommendations using the trained agent."""
        if not self.is_ready:
            print("⚠️ Learning system not ready, using fallback recommendations")
            return self._get_fallback_recommendations(limit)
        
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
    
    def _get_fallback_recommendations(self, limit: int) -> list:
        """Fallback recommendations when system is not ready."""
        if self.catalog:
            # Use existing catalog if available
            popular_products = self.catalog.products_df.nlargest(limit, 'popularity')
            return [
                {
                    'product_id': idx,
                    'category_name': row['category_name'],
                    'price': row['price'],
                    'popularity': row['popularity'],
                    'quality': row['quality']
                }
                for idx, row in popular_products.iterrows()
            ]
        else:
            # Fallback stub data
            return [
                {
                    'product_id': i,
                    'category_name': f'Category_{i % 5}',
                    'price': 10.0 + i,
                    'popularity': 0.5,
                    'quality': 0.5
                }
                for i in range(limit)
            ]
    
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
                
                # print(f"🧠 Agent learned from '{action}' action (reward: {reward})")
            
        except Exception as e:
            print(f"⚠️ Learning update failed: {e}")
    
    def get_learning_stats(self) -> dict:
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
    
    async def shutdown(self):
        """Shutdown the learning manager."""
        self.is_ready = False
        print("✅ Learning manager shutdown completed")

    """
    для репорта

    The algorithm updates user preferences based on their interactions with products, 
    considering three key factors: action type (purchase > add_to_cart > like > view), 
    time relevance (newer actions matter more via exponential decay), and reward value. 
    Category preferences are calculated by analyzing which product categories the user 
    interacts with most frequently and positively. Style preferences are derived from 
    the style_vector of products the user engages with, identifying patterns in product 
    characteristics they prefer. Updates are applied gradually with a learning rate of 0.1 
    to ensure stable evolution of preferences without sudden jumps in recommendations.
    """
    async def update_user_preferences(self, db, user_id: int, product_id: int, action: str):
        try:
            thirty_days_ago = datetime.now() - timedelta(days=30)
            
            stmt = select(UserAction).where(
                UserAction.user_id == user_id,
                UserAction.action_timestamp >= thirty_days_ago
            ).order_by(UserAction.action_timestamp.desc())
            
            result = await db.execute(stmt)
            recent_actions = result.scalars().all()
            
            if not recent_actions:
                return
            
            product_stmt = select(Product).where(Product.product_id == product_id)
            product_result = await db.execute(product_stmt)
            product = product_result.scalar_one_or_none()
            
            if not product:
                return
            
            action_weights = self._get_action_weights()
            time_decay_factor = self._calculate_time_decay(recent_actions)
            
            category_prefs = await self._calculate_category_preferences(
                db, recent_actions, action_weights, time_decay_factor
            )
            
            style_prefs = await self._calculate_style_preferences(
                db, recent_actions, action_weights, time_decay_factor, product
            )
            
            user_stmt = select(User).where(User.user_id == user_id)
            user_result = await db.execute(user_stmt)
            user = user_result.scalar_one_or_none()
            
            if user:
                # Smooth upgrade with learning rate
                learning_rate = 0.1
                
                if user.category_preferences:
                    # Combining old and new preferences
                    for category, score in category_prefs.items():
                        old_score = user.category_preferences.get(category, 0.5)
                        user.category_preferences[category] = (
                            (1 - learning_rate) * old_score + learning_rate * score
                        )
                else:
                    user.category_preferences = category_prefs
                
                if user.style_preferences:
                    # Combining style preferences
                    for style, score in style_prefs.items():
                        old_score = user.style_preferences.get(style, 0.5)
                        user.style_preferences[style] = (
                            (1 - learning_rate) * old_score + learning_rate * score
                        )
                else:
                    user.style_preferences = style_prefs
                
                await db.commit()
                
        except Exception as e:
            print(f"Error updating user preferences: {e}")
            await db.rollback()
    
    def _get_action_weights(self) -> Dict[str, float]:
        """Returns weights for different types of actions"""
        return {
            'view': 0.3,
            'like': 1.0,
            'add_to_cart': 2.0,
            'purchase': 3.0,
            'share': 1.5,
            'dislike': -1.0,
            'close_immediately': -0.5,
            'report': -2.0,
            'remove_from_cart': -0.3
        }
    
    def _calculate_time_decay(self, actions: List[UserAction]) -> List[float]:
        """Calculates a time deck for actions (new actions are more important)"""
        if not actions:
            return []
        
        latest_time = max(action.action_timestamp for action in actions)
        time_decays = []
        
        for action in actions:
            hours_diff = (latest_time - action.action_timestamp).total_seconds() / 3600
            # Exponential deck: actions in the last 24 hours have a weight of ~1.0
            decay = np.exp(-hours_diff / 24.0)
            time_decays.append(decay)
        
        return time_decays
    
    async def _calculate_category_preferences(self,
                                           db, 
                                           actions: List[UserAction], 
                                           action_weights: Dict[str, float],
                                           time_decays: List[float]) -> Dict[str, float]:
        """Calculates preferences by category"""
        category_scores = defaultdict(float)
        category_weights = defaultdict(float)
        
        for i, action in enumerate(actions):
            if action.product_id is None:
                continue
                
            product_stmt = select(Product).where(Product.product_id == action.product_id)
            result = await db.execute(product_stmt)
            product = result.scalar_one_or_none()
            
            if not product:
                continue
            
            category = product.category_name
            action_weight = action_weights.get(action.action_type, 0.1)
            time_decay = time_decays[i] if i < len(time_decays) else 1.0
            
            total_weight = action_weight * time_decay * (1 + action.reward * 0.1)
            
            category_scores[category] += total_weight
            category_weights[category] += abs(total_weight)
        
        # Normalize scores from 0 to 1
        category_preferences = {}
        all_categories = ['Electronics', 'Clothing', 'Books', 'Home & Garden', 
                         'Sports', 'Beauty', 'Toys', 'Automotive', 'Health', 'Food']
        
        for category in all_categories:
            if category_weights[category] > 0:
                # Convert to the range [0, 1] using sigmoids
                raw_score = category_scores[category] / category_weights[category]
                normalized_score = 1 / (1 + np.exp(-raw_score * 3))  # Reinforcing the differences
                category_preferences[category] = float(normalized_score)
            else:
                category_preferences[category] = 0.5
        
        return category_preferences
    
    async def _calculate_style_preferences(self,
                                         db,
                                         actions: List[UserAction],
                                         action_weights: Dict[str, float],
                                         time_decays: List[float],
                                         current_product: Product) -> Dict[str, float]:
        """Calculates style preferences based on the style_vector of products"""
        style_scores = defaultdict(float)
        style_weights = defaultdict(float)
        
        for i, action in enumerate(actions):
            if action.product_id is None:
                continue
                
            product_stmt = select(Product).where(Product.product_id == action.product_id)
            result = await db.execute(product_stmt)
            product = result.scalar_one_or_none()
            
            if not product or not product.style_vector:
                continue
            
            action_weight = action_weights.get(action.action_type, 0.1)
            time_decay = time_decays[i] if i < len(time_decays) else 1.0
            total_weight = action_weight * time_decay * (1 + action.reward * 0.1)
            
            for style_key, style_value in product.style_vector.items():
                if isinstance(style_value, (int, float)):
                    style_scores[style_key] += style_value * total_weight
                    style_weights[style_key] += abs(total_weight)
        
        # Adding the influence of the current product with more weight
        if current_product and current_product.style_vector:
            current_weight = 2.0  # More weight for the current interaction
            for style_key, style_value in current_product.style_vector.items():
                if isinstance(style_value, (int, float)):
                    style_scores[style_key] += style_value * current_weight
                    style_weights[style_key] += current_weight
        
        # Normalize style preferences
        style_preferences = {}
        for style_key in style_scores.keys():
            if style_weights[style_key] > 0:
                raw_score = style_scores[style_key] / style_weights[style_key]
                # Limit the range and normalize
                normalized_score = max(0.0, min(1.0, (raw_score + 1) / 2))
                style_preferences[style_key] = float(normalized_score)
        
        # If there are no style preferences, we use neutral values
        if not style_preferences and current_product.style_vector:
            for style_key, style_value in current_product.style_vector.items():
                if isinstance(style_value, (int, float)):
                    style_preferences[style_key] = 0.5
        
        return style_preferences
    
    def get_user_preference_vector(self, user: User) -> np.ndarray:
        """Создает вектор предпочтений для использования в рекомендациях"""
        if not user.category_preferences or not user.style_preferences:
            return np.zeros(50)
        
        category_vector = []
        for category in ['Electronics', 'Clothing', 'Books', 'Home & Garden', 
                        'Sports', 'Beauty', 'Toys', 'Automotive', 'Health', 'Food']:
            category_vector.append(user.category_preferences.get(category, 0.5))
        
        style_vector = list(user.style_preferences.values())
        
        combined_vector = category_vector + style_vector
        if len(combined_vector) < 50:
            combined_vector.extend([0.5] * (50 - len(combined_vector)))
        elif len(combined_vector) > 50:
            combined_vector = combined_vector[:50]
        
        return np.array(combined_vector)


import atexit

# Global instance
learning_manager = GlobalLearningManager()

_initialization_started = False
_initialization_task = None

async def initialize_learning_system():
    """Initialize the learning system on startup"""
    global _initialization_started, _initialization_task
    
    if _initialization_started:
        if _initialization_task:
            await _initialization_task
        return learning_manager.is_ready
    
    if not learning_manager.is_ready:
        _initialization_started = True
        print("🔄 Starting automatic learning system initialization...")
        try:
            success = await learning_manager.initialize_system(n_products=500, n_users=100)
            return success
        except Exception as e:
            print(f"❌ Automatic initialization failed: {e}")
            _initialization_started = False
            return False
    
    return True

async def ensure_learning_system_ready():
    """Ensure learning system is ready, initialize if needed"""
    if learning_manager.is_ready:
        return True
    
    return await initialize_learning_system()
