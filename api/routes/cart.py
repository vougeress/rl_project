"""
Shopping cart and product action API routes.
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any
from api.models.schemas import CartItemCreate, CartResponse, OrderResponse, UserActionCreate
from api.services.cart_service import CartService
from api.core.database import get_db

router = APIRouter(tags=["cart", "products"])

@router.post("/{user_id}/actions", response_model=Dict[str, Any])
async def product_action(
    user_id: int,
    action_data: UserActionCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Handle user actions on products (like, dislike, report, view, share, close_immediately).

    - **user_id**: User identifier
    - **action_data**: Action details including product_id and action_type
    """
    try:
        from api.services.recommendation_service import RecommendationService
        recommendation_service = RecommendationService()

        if not action_data.product_id:
            raise ValueError("product_id is required for product actions")

        result = await recommendation_service.process_user_action(
            db, user_id, action_data.product_id, action_data.action_type, action_data.experiment_id
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process action: {str(e)}")

@router.post("/{user_id}/items", response_model=Dict[str, Any])
async def add_to_cart(
    user_id: int,
    cart_item: CartItemCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Add product to user's shopping cart.

    - **user_id**: User identifier
    - **cart_item**: Cart item details including product_id and quantity
    """
    try:
        cart_service = CartService()
        result = await cart_service.add_to_cart(db, user_id, cart_item)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to add to cart: {str(e)}")

@router.get("/{user_id}", response_model=CartResponse)
async def get_shopping_cart(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get user's shopping cart with all items and totals.

    - **user_id**: User identifier
    """
    try:
        cart_service = CartService()
        return await cart_service.get_cart(db, user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get cart: {str(e)}")

@router.put("/{user_id}/items/{cart_item_id}", response_model=Dict[str, Any])
async def update_cart_item(
    user_id: int,
    cart_item_id: int,
    quantity: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Update cart item quantity.

    - **user_id**: User identifier
    - **cart_item_id**: Cart item identifier
    - **quantity**: New quantity (must be at least 1)
    """
    try:
        if quantity < 1:
            raise ValueError("Quantity must be at least 1")

        cart_service = CartService()
        result = await cart_service.update_cart_item(db, user_id, cart_item_id, quantity)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update cart item: {str(e)}")

@router.delete("/{user_id}/items/{cart_item_id}", response_model=Dict[str, Any])
async def remove_from_cart(
    user_id: int,
    cart_item_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Remove item from cart.

    - **user_id**: User identifier
    - **cart_item_id**: Cart item identifier
    """
    try:
        cart_service = CartService()
        result = await cart_service.remove_from_cart(db, user_id, cart_item_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to remove from cart: {str(e)}")

@router.delete("/{user_id}/clear", response_model=Dict[str, Any])
async def clear_cart(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Clear user's entire cart.

    - **user_id**: User identifier
    """
    try:
        cart_service = CartService()
        result = await cart_service.clear_cart(db, user_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to clear cart: {str(e)}")

@router.post("/{user_id}/order", response_model=OrderResponse)
async def place_order(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Place an order from the shopping cart.

    - **user_id**: User identifier
    """
    try:
        cart_service = CartService()
        return await cart_service.place_order(db, user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to place order: {str(e)}")

@router.get("/{user_id}/orders", response_model=List[Dict[str, Any]])
async def get_order_history(
    user_id: int,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    Get user's order history.

    - **user_id**: User identifier
    - **limit**: Number of orders to return (default: 10)
    """
    try:
        if limit > 50:
            limit = 50
        cart_service = CartService()
        return await cart_service.get_order_history(db, user_id, limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get order history: {str(e)}")

@router.get("/learning/stats")
async def get_learning_statistics():
    """Get learning statistics from the global learning manager."""
    try:
        from api.core.learning_manager import learning_manager
        stats = learning_manager.get_learning_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))