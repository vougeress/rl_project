"""
User service for handling user registration and management.
"""

import uuid
from typing import Optional
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# from api.core.learning_manager import learning_manager
from api.models.database_models import User, UserSession
from api.models.schemas import UserRegistration, UserRegistrationResponse, UserSessionResponse, UserSessionCreate


class UserService:
    def __init__(self):
        pass    
    async def register_user(self, db: AsyncSession, user_data: UserRegistration, experiment_id: Optional[str] = None) -> UserRegistrationResponse:
        """Register a new user with dataset medians for missing fields."""
        try:
            stmt = select(User)
            result = await db.execute(stmt)
            existing_users = result.scalars().all()
            
            if existing_users:
                price_sensitivities = [u.price_sensitivity for u in existing_users]
                quality_sensitivities = [u.quality_sensitivity for u in existing_users]
                exploration_tendencies = [u.exploration_tendency for u in existing_users]
                budget_multipliers = [u.budget_multiplier for u in existing_users]
                
                medians = {
                    'price_sensitivity': float(sum(price_sensitivities) / len(price_sensitivities)),
                    'quality_sensitivity': float(sum(quality_sensitivities) / len(quality_sensitivities)),
                    'exploration_tendency': float(sum(exploration_tendencies) / len(exploration_tendencies)),
                    'budget_multiplier': float(sum(budget_multipliers) / len(budget_multipliers))
                }
            else:
                medians = {
                    'price_sensitivity': 0.5,
                    'quality_sensitivity': 0.5,
                    'exploration_tendency': 0.5,
                    'budget_multiplier': 1.0
                }

            if user_data.age < 25:
                income_level = 'low'
            elif user_data.age < 45:
                income_level = 'medium'
            else:
                income_level = 'high'

            new_user = User(
                name=user_data.name,
                age=user_data.age,
                income_level=income_level,
                price_sensitivity=medians['price_sensitivity'],
                quality_sensitivity=medians['quality_sensitivity'],
                exploration_tendency=medians['exploration_tendency'],
                budget_multiplier=medians['budget_multiplier'],
                category_preferences={},
                style_preferences={},
                state_vector={},
                registered_at=datetime.now()
            )
            
            db.add(new_user)
            await db.flush()

            session = await self._ensure_user_session(db, new_user.user_id, experiment_id)
            
            await db.commit()
            
            return UserRegistrationResponse(
                user_id=new_user.user_id,
                name=user_data.name,
                age=user_data.age,
                income_level=income_level,
                profile_completed_with_medians=medians,
                message='User registered successfully'
            )
        
        except Exception as e:
            await db.rollback()
            raise ValueError(f"Failed to register user: {str(e)}")
    
    async def _ensure_user_session(self, db: AsyncSession, user_id: int, experiment_id: Optional[str] = None) -> UserSessionResponse:
        """Ensure user has an active session, create if not exists."""
        stmt = select(UserSession).where(
            UserSession.user_id == user_id,
            UserSession.end_time.is_(None)
        )
        result = await db.execute(stmt)
        active_session = result.scalar_one_or_none()
        
        if active_session:
            return await self._session_to_response(active_session)
        
        session_id = str(uuid.uuid4())
        user_session = UserSession(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now(),
            experiment_id=experiment_id
        )
        
        db.add(user_session)
        await db.flush()
        
        return await self._session_to_response(user_session)
    
    async def update_session_on_action(self, db: AsyncSession, session_id: str, reward: float) -> bool:
        """Update session stats when user performs action."""
        try:
            stmt = select(UserSession).where(UserSession.session_id == session_id)
            result = await db.execute(stmt)
            session = result.scalar_one_or_none()
            
            if session and session.end_time is None:  # Только для активных сессий
                session.actions_count = (session.actions_count or 0) + 1
                session.total_reward = (session.total_reward or 0.0) + reward
                await db.commit()
                return True
            return False
        except Exception as e:
            await db.rollback()
            print(f"Error updating session {session_id}: {e}")
            return False
    
    async def end_session(self, db: AsyncSession, session_id: str) -> bool:
        """End user session."""
        try:
            stmt = select(UserSession).where(UserSession.session_id == session_id)
            result = await db.execute(stmt)
            session = result.scalar_one_or_none()

            if session and session.end_time is None:
                session.end_time = datetime.now()
                if session.start_time:
                    session.session_length = int((session.end_time - session.start_time).total_seconds())
                await db.commit()
                return True
            return False
        except Exception as e:
            await db.rollback()
            print(f"Error ending session {session_id}: {e}")
            return False
    
    async def _session_to_response(self, session: UserSession) -> UserSessionResponse:
        """Convert UserSession model to response schema."""
        return UserSessionResponse(
            session_id=session.session_id,
            user_id=session.user_id,
            start_time=session.start_time,
            end_time=session.end_time,
            session_length=session.session_length or 0,
            total_reward=session.total_reward or 0.0,
            actions_count=session.actions_count or 0,
            experiment_id=session.experiment_id,
            is_active=session.end_time is None
        )
    
    async def get_user_session(self, db: AsyncSession, user_id: int) -> Optional[UserSessionResponse]:
        """Get user's active session."""
        stmt = select(UserSession).where(
            UserSession.user_id == user_id,
            UserSession.end_time.is_(None)
        )
        result = await db.execute(stmt)
        session = result.scalar_one_or_none()
        
        return await self._session_to_response(session) if session else None
    
    async def validate_user(self, db: AsyncSession, user_id: int) -> bool:
        """Validate if user exists in database."""
        try:
            stmt = select(User).where(User.user_id == user_id)
            result = await db.execute(stmt)
            user = result.scalar_one_or_none()
            return user is not None
        except Exception:
            return False
    
    async def get_user_by_id(self, db: AsyncSession, user_id: int) -> Optional[User]:
        """Get user by ID from database."""
        stmt = select(User).where(User.user_id == user_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()