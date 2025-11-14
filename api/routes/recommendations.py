"""
Recommendation API routes.
"""

from fastapi import APIRouter, HTTPException
from api.models.schemas import RecommendationsResponse
from api.services.recommendation_service import RecommendationService

router = APIRouter(prefix="/recommendations", tags=["recommendations"])
recommendation_service = RecommendationService()


@router.get("/{user_id}", response_model=RecommendationsResponse)
async def get_general_recommendations(user_id: int, limit: int = 20):
    """Get general product recommendations for a user (20 products by default)."""
    try:
        return recommendation_service.get_recommendations(user_id, limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))