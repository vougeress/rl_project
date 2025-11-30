"""
User API routes.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from api.models.schemas import UserRegistration, UserRegistrationResponse, UserInfo
from api.services.user_service import UserService
from api.core.database import get_db

router = APIRouter(prefix="/user", tags=["users"])
    

@router.post("/register", response_model=UserRegistrationResponse)
async def register_user(
    user_data: UserRegistration,
    experiment_id: Optional[str] = Query(None, description="Optional experiment ID"),
    db: AsyncSession  = Depends(get_db)
):
    try:
        user_service = UserService()
        return await user_service.register_user(db, user_data, experiment_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register user: {str(e)}")


@router.get("/{user_id}", response_model=UserInfo)
async def get_user_info(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    try:
        user_service = UserService()
        user = await user_service.get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserInfo(
            user_id=user.user_id,
            name=user.name,
            age=user.age,
            income_level=user.income_level,
            price_sensitivity=user.price_sensitivity,
            quality_sensitivity=user.quality_sensitivity,
            exploration_tendency=user.exploration_tendency
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user info: {str(e)}")


@router.post("/{user_id}/sessions/{session_id}/end", response_model=dict)
async def end_user_session(
    user_id: int,
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        user_service = UserService()

        if not await user_service.validate_user(db, user_id):
            raise HTTPException(status_code=404, detail="User not found")
        
        session = await user_service.get_user_session(db, user_id)
        if not session or session.session_id != session_id:
            raise HTTPException(status_code=403, detail="Session does not belong to user")
        
        success = await user_service.end_session(db, session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found or already ended")
        
        return {"message": "Session ended successfully", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}")