"""
Recommendation service for handling product recommendations.
"""

from typing import Any, List, Dict, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from api.core.learning_manager import learning_manager
from api.models.schemas import ProductInfo, RecommendationsResponse
from api.models.database_models import Product, UserAction


class RecommendationService:
    def __init__(self):
        pass

    async def get_recommendations(self, db: AsyncSession, user_id: int, limit: int = 20) -> RecommendationsResponse:
        """Get product recommendations using the global learning agent."""
        try:
            from api.services.user_service import UserService
            user_service = UserService()
            if not await user_service.validate_user(db, user_id):
                raise ValueError(f"User {user_id} does not exist")

            if not learning_manager.is_ready:
                print("🔄 Learning system not ready in recommendation service, using fallback")
                recommendations_data = await self._get_fallback_recommendations(db, limit)
            else:
                # Get recommendations from global learning manager
                recommendations_data = await learning_manager.get_recommendations(user_id, limit)

            if not recommendations_data:
                recommendations_data = await self._get_fallback_recommendations(
                    db, limit)

            # Convert to ProductInfo objects
            recommendations = []
            for product_info in recommendations_data:
                recommendations.append(ProductInfo(
                    product_id=product_info['product_id'],
                    product_name=product_info.get('product_name', f"Product {product_info['product_id']}"),
                    name_format=product_info.get('name_format'),
                    category_name=product_info['category_name'],
                    price=product_info['price'],
                    popularity=product_info['popularity'],
                    quality=product_info['quality']
                ))

            return RecommendationsResponse(
                products=recommendations,
                user_id=user_id,
                total_count=len(recommendations)
            )

        except Exception as e:
            raise ValueError(f"Failed to get recommendations: {str(e)}")

    async def get_product_info(self, db: AsyncSession, product_id: int) -> Dict:
        """Get product information from database."""
        try:
            stmt = select(Product).where(Product.product_id == product_id)
            result = await db.execute(stmt)
            product = result.scalar_one_or_none()

            if not product:
                raise ValueError(f"Product {product_id} not found")

            return {
                'product_id': product.product_id,
                'product_name': product.product_name,
                'name_format': product.name_format,
                'category_id': product.category_id,
                'category_name': product.category_name,
                'price': product.price,
                'popularity': product.popularity,
                'quality': product.quality,
                'style_vector': product.style_vector,
                'created_at': product.created_at
            }
        except Exception as e:
            raise ValueError(f"Failed to get product info: {str(e)}")

    async def _get_fallback_recommendations(self, db: AsyncSession, limit: int = 20) -> List[Dict]:
        """Fallback recommendations from database when learning system is not ready."""
        try:
            stmt = select(Product).order_by(Product.popularity.desc()).limit(limit)
            result = await db.execute(stmt)
            products = result.scalars().all()

            recommendations = []
            for product in products:
                recommendations.append({
                    'product_id': product.product_id,
                    'product_name': product.product_name,
                    'name_format': product.name_format,
                    'category_name': product.category_name,
                    'price': product.price,
                    'popularity': product.popularity,
                    'quality': product.quality
                })

            return recommendations
        except Exception as e:
            return []

    async def process_user_action(self, db: AsyncSession, user_id: int, product_id: int, action: str, 
                                  experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """Process user action and update learning system."""
        try:
            from api.services.user_service import UserService
            user_service = UserService()

            if not await user_service.validate_user(db, user_id):
                raise ValueError(f"User {user_id} does not exist")

            stmt = select(Product).where(Product.product_id == product_id)
            result = await db.execute(stmt)
            product = result.scalar_one_or_none()

            if not product:
                raise ValueError(f"Product {product_id} not found")

            user_session = await user_service.get_user_session(db, user_id)
            if not user_session:
                user_session = await user_service._ensure_user_session(
                    db, user_id, experiment_id)

            actual_session_id = user_session.session_id

            reward = self._calculate_reward(action, product)
            
            session_time = int((datetime.now() - user_session.start_time).total_seconds())

            user_action = UserAction(
                user_id=user_id,
                product_id=product_id,
                action_type=action,
                reward=reward,
                session_time=session_time,
                session_id=actual_session_id,
                experiment_id=experiment_id,
                action_timestamp=datetime.now()
            )

            db.add(user_action)

            await user_service.update_session_on_action(db, actual_session_id, reward, auto_commit=False)
            session_state = await user_service.update_user_state_vector(
                db, user_id, actual_session_id, action, reward
            )

            if learning_manager.is_ready and action not in ['add_to_cart', 'remove_from_cart', 'purchase']:
                session_context = {
                    'current_session_actions': session_state.get('current_session_actions', 0),
                    'current_session_reward': session_state.get('current_session_reward', 0.0),
                    'average_reward': session_state.get('average_reward', reward)
                }
                learning_manager.learn_from_action(
                    user_id,
                    product_id,
                    action,
                    reward,
                    session_context=session_context
                )
                await learning_manager.update_user_preferences(db, user_id, product_id, action)
            
            await db.commit()

            return {
                'success': True,
                'action_id': user_action.action_id,
                'session_id': actual_session_id,
                'reward': user_action.reward,
                'message': f'Action {action} processed successfully'
            }

        except Exception as e:
            await db.rollback()
            raise ValueError(f"Failed to process user action: {str(e)}")

    def _calculate_reward(self, action: str, product: Product) -> float:
        """Calculate reward based on action type and product properties."""
        reward_map = {
            'like': 1.0,
            'add_to_cart': 3.0,
            'purchase': 8.0,
            'dislike': -0.5,
            'report': -2.0,
            'view': 0.3,
            'share': 4.0,
            'close_immediately': -0.5,
            'remove_from_cart': -0.5
        }

        base_reward = reward_map.get(action, 0.0)

        # Модифицируем награду на основе качества и популярности продукта
        quality_bonus = product.quality * 0.5
        popularity_bonus = product.popularity * 0.3

        return base_reward + quality_bonus + popularity_bonus
