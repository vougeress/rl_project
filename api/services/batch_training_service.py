"""
Service for batch training operations - bulk user creation and actions simulation.
"""

import asyncio
import random
import uuid
from datetime import datetime
from typing import List, Dict, Any
from api.core.learning_manager import GlobalLearningManager
from api.services.user_service import UserService
from api.services.recommendation_service import RecommendationService
from api.models.schemas import UserRegistration
from src.data_generation import generate_synthetic_data, ProductCatalog, UserSimulator


class BatchTrainingService:
    def __init__(self):
        self.learning_manager = GlobalLearningManager()
        self.user_service = UserService()
        self.recommendation_service = RecommendationService()
        self.catalog = None
        self.simulator = None
        self.simulation_stats = {
            'total_users': 0,
            'total_actions': 0,
            'learning_episodes': 0,
            'start_time': None
        }
    
    async def bulk_register_users(self, count: int) -> Dict[str, Any]:
        """Register multiple users with random profiles."""
        user_ids = []
        
        for _ in range(count):
            # Generate random user data
            name = f"User_{random.randint(1000, 9999)}"
            age = random.randint(18, 65)
            
            # Create UserRegistration object
            user_registration = UserRegistration(name=name, age=age)
            
            # Register user
            user_data = self.user_service.register_user(user_registration)
            user_ids.append(str(user_data.user_id))
        
        self.simulation_stats['total_users'] += count
        
        return {
            'message': f'Successfully registered {count} users',
            'users_created': count,
            'user_ids': user_ids
        }
    
    async def process_bulk_actions(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process multiple user actions for training."""
        processed_count = 0
        learning_updates = 0
        
        for action_data in actions:
            try:
                user_id = action_data['user_id']
                product_id = action_data['product_id']
                action = action_data['action']
                
                # Process the action through recommendation service
                result = await self.recommendation_service.process_user_action(
                    user_id, product_id, action
                )
                
                processed_count += 1
                learning_updates += 1
                
            except Exception as e:
                print(f"Error processing action: {e}")
                continue
        
        self.simulation_stats['total_actions'] += processed_count
        self.simulation_stats['learning_episodes'] += learning_updates
        
        return {
            'message': f'Processed {processed_count} actions',
            'actions_processed': processed_count,
            'learning_updates': learning_updates
        }
    
    async def simulate_user_behavior(self, num_users: int, actions_per_user: int, 
                                   simulation_speed: float = 1.0) -> Dict[str, Any]:
        """Simulate realistic user behavior for training."""
        simulation_id = str(uuid.uuid4())[:8]
        
        # First, register users
        bulk_users = await self.bulk_register_users(num_users)
        user_ids = [int(uid) for uid in bulk_users['user_ids']]
        
        # Initialize data if not exists
        if self.catalog is None or self.simulator is None:
            self.catalog, self.simulator = generate_synthetic_data(500, 100)
        
        total_actions = 0
        actions_batch = []
        
        # Simulate actions for each user
        for user_id in user_ids:
            for _ in range(actions_per_user):
                # Get recommendations for user
                try:
                    recommendations = await self.recommendation_service.get_recommendations(user_id)
                    
                    if recommendations['products']:
                        # Choose a random product from recommendations
                        product = random.choice(recommendations['products'])
                        product_id = product['product_id']
                        
                        # Simulate realistic action probabilities
                        action_weights = {
                            'like': 0.4,
                            'dislike': 0.2,
                            'add_to_cart': 0.3,
                            'report': 0.1
                        }
                        
                        action = random.choices(
                            list(action_weights.keys()),
                            weights=list(action_weights.values())
                        )[0]
                        
                        actions_batch.append({
                            'user_id': user_id,
                            'product_id': product_id,
                            'action': action
                        })
                        total_actions += 1
                        
                        # Process in batches to avoid memory issues
                        if len(actions_batch) >= 100:
                            await self.process_bulk_actions(actions_batch)
                            actions_batch = []
                            
                            # Simulate delay based on speed
                            await asyncio.sleep(0.1 / simulation_speed)
                
                except Exception as e:
                    print(f"Error in simulation for user {user_id}: {e}")
                    continue
        
        # Process remaining actions
        if actions_batch:
            await self.process_bulk_actions(actions_batch)
        
        estimated_duration = (total_actions * 0.1) / simulation_speed
        
        return {
            'message': f'Simulation {simulation_id} completed',
            'simulation_id': simulation_id,
            'users_created': num_users,
            'total_actions': total_actions,
            'estimated_duration': estimated_duration
        }
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training statistics."""
        try:
            # Get learning stats from the manager
            learning_stats = self.learning_manager.get_learning_stats()
            
            # Get agent-specific stats
            current_epsilon = 0.0
            average_reward = learning_stats.get('avg_reward', 0.0)
            
            if hasattr(self.learning_manager, 'agent') and self.learning_manager.agent:
                agent = self.learning_manager.agent
                if hasattr(agent, 'epsilon'):
                    current_epsilon = agent.epsilon
            
            return {
                'total_users': self.simulation_stats['total_users'],
                'total_actions': self.simulation_stats['total_actions'],
                'learning_episodes': learning_stats.get('total_episodes', 0),
                'current_epsilon': current_epsilon,
                'average_reward': average_reward,
                'last_update': datetime.now().isoformat()
            }
        except Exception as e:
            # Fallback response
            return {
                'total_users': self.simulation_stats['total_users'],
                'total_actions': self.simulation_stats['total_actions'],
                'learning_episodes': 0,
                'current_epsilon': 0.0,
                'average_reward': 0.0,
                'last_update': datetime.now().isoformat(),
                'error': str(e)
            }
    
    async def reset_training_stats(self):
        """Reset training statistics."""
        self.simulation_stats = {
            'total_users': 0,
            'total_actions': 0,
            'learning_episodes': 0,
            'start_time': datetime.now()
        }
        
        # Reset agent if needed
        if hasattr(self.learning_manager, 'agent') and self.learning_manager.agent:
            agent = self.learning_manager.agent
            if hasattr(agent, 'reset'):
                agent.reset()