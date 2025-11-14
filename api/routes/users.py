"""
User API routes.
"""

from fastapi import APIRouter, HTTPException
from api.models.schemas import UserRegistration, UserRegistrationResponse
from api.services.user_service import UserService

router = APIRouter(prefix="/user", tags=["users"])
user_service = UserService()


@router.post("/register", response_model=UserRegistrationResponse)
async def register_user(user_data: UserRegistration):
    """Register a new user with dataset medians for missing fields."""
    try:
        return user_service.register_user(user_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))