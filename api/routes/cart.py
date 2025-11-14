"""
Shopping cart and product action API routes.
"""

from fastapi import APIRouter, HTTPException
from api.models.schemas import UserAction, ProductActionResponse, CartResponse, OrderResponse
from api.services.cart_service import CartService

router = APIRouter(tags=["cart", "products"])
cart_service = CartService()


@router.post("/product/action", response_model=ProductActionResponse)
async def product_action(action_data: UserAction):
    """Handle user actions on products (like, dislike, report, add_to_cart)."""
    try:
        return cart_service.handle_product_action(action_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cart/{user_id}", response_model=CartResponse)
async def get_shopping_cart(user_id: int):
    """Get user's shopping cart."""
    try:
        return cart_service.get_shopping_cart(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cart/{user_id}/order", response_model=OrderResponse)
async def place_order(user_id: int):
    """Place an order from the shopping cart."""
    try:
        return cart_service.place_order(user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/learning/stats")
async def get_learning_statistics():
    """Get learning statistics from the global learning manager."""
    try:
        from api.core.learning_manager import learning_manager
        stats = learning_manager.get_learning_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))