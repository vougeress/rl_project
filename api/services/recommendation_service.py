"""
Recommendation service for handling product recommendations.
"""

import uuid
from typing import List, Dict, Optional
from datetime import datetime

from api.core.learning_manager import learning_manager
from api.models.schemas import ProductInfo, RecommendationsResponse


class RecommendationService:
    def __init__(self):
        pass
    
    def get_recommendations(self, user_id: int, limit: int = 20) -> RecommendationsResponse:
        """Get product recommendations using the global learning agent."""
        try:
            # Get recommendations from global learning manager
            recommendations_data = learning_manager.get_recommendations(user_id, limit)
            
            # Convert to ProductInfo objects
            recommendations = []
            for product_info in recommendations_data:
                recommendations.append(ProductInfo(
                    product_id=product_info['product_id'],
                    category=product_info['category'],
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
    
    def get_product_info(self, product_id: int) -> Dict:
        """Get product information."""
        if not learning_manager.is_ready:
            learning_manager.initialize_system()
        
        if product_id >= learning_manager.catalog.n_products:
            raise ValueError(f"Product ID must be less than {learning_manager.catalog.n_products}")
        
        return learning_manager.catalog.get_product_info(product_id)