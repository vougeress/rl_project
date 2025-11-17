"""
Recommendation API routes.
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from api.models.schemas import RecommendationsResponse
from api.services.recommendation_service import RecommendationService
from api.core.database import get_db
from api.models.schemas import UserActionCreate

router = APIRouter(prefix="/recommendations", tags=["recommendations"])    

@router.get("/{user_id}", response_model=RecommendationsResponse)
async def get_general_recommendations(
    user_id: int, 
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """
    Get general product recommendations for a user.
    
    - **user_id**: User identifier
    - **limit**: Number of recommendations (default: 20, max: 50)
    """
    try:
        if limit > 50:
            limit = 50
        recommendation_service = RecommendationService()
        return await recommendation_service.get_recommendations(db, user_id, limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")
    

@router.post("/{user_id}/action", response_model=dict)
async def process_user_action(
    user_id: int,
    action_data: UserActionCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Process user action and update recommendations.
    
    - **user_id**: User identifier
    - **action_data**: Action details including product_id and action_type
    """
    try:
        product_id = action_data.product_id
        action_type = action_data.action_type
        experiment_id = action_data.experiment_id
        
        if not product_id or not action_type:
            raise HTTPException(status_code=400, detail="product_id and action_type are required")
        
        recommendation_service = RecommendationService()
        result = await recommendation_service.process_user_action(
            db, user_id, product_id, action_type, experiment_id
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process action: {str(e)}")


@router.get("/products/{product_id}", response_model=dict)
async def get_product_info(
    product_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed information about a product.
    
    - **product_id**: Product identifier
    """
    try:
        recommendation_service = RecommendationService()
        product_info = await recommendation_service.get_product_info(db, product_id)
        return product_info
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get product info: {str(e)}")