"""
User service for handling user registration and management.
"""

import uuid
from typing import Dict
from datetime import datetime

from api.core.learning_manager import learning_manager
from api.models.schemas import UserRegistration, UserRegistrationResponse


class UserService:
    def __init__(self):
        self.user_sessions = {}
        self.next_user_id = None  # Will be initialized when learning system is ready
    
    def register_user(self, user_data: UserRegistration) -> UserRegistrationResponse:
        """Register a new user with dataset medians for missing fields."""
        # Ensure learning system is ready
        if not learning_manager.is_ready:
            learning_manager.initialize_system()
        
        # Calculate dataset medians
        users_df = learning_manager.simulator.users_df
        medians = {
            'price_sensitivity': float(users_df['price_sensitivity'].median()),
            'quality_sensitivity': float(users_df['quality_sensitivity'].median()),
            'exploration_tendency': float(users_df['exploration_tendency'].median()),
            'budget_multiplier': float(users_df['budget_multiplier'].median())
        }
        
        # Determine income level based on age (simple heuristic)
        if user_data.age < 25:
            income_level = 'low'
        elif user_data.age < 45:
            income_level = 'medium'
        else:
            income_level = 'high'
        
        # Initialize next_user_id if not set
        if self.next_user_id is None:
            self.next_user_id = len(users_df)
        
        # Create new unique user ID
        new_user_id = self.next_user_id
        self.next_user_id += 1
        
        # Store user session
        self.user_sessions[new_user_id] = {
            'name': user_data.name,
            'age': user_data.age,
            'income_level': income_level,
            'registered_at': datetime.now(),
            **medians
        }
        
        return UserRegistrationResponse(
            user_id=new_user_id,
            name=user_data.name,
            age=user_data.age,
            income_level=income_level,
            profile_completed_with_medians=medians,
            message='Пользователь успешно зарегистрирован'
        )
    
    def get_user_session(self, user_id: int) -> Dict:
        """Get user session data."""
        return self.user_sessions.get(user_id, {})
    
    def validate_user(self, user_id: int) -> bool:
        """Validate if user exists."""
        if not learning_manager.is_ready:
            return False
        return user_id < learning_manager.simulator.n_users or user_id in self.user_sessions