"""
Shopping cart service for handling cart operations and orders.
"""

from typing import Dict, List, Any
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from api.models.database_models import CartItem, Product, Order, UserAction
from api.models.schemas import CartItemCreate, CartResponse, OrderResponse, ProductInfo
from api.core.learning_manager import learning_manager


class CartService:
    def __init__(self):
        pass
    
    async def add_to_cart(self, db: AsyncSession, user_id: int, cart_data: CartItemCreate) -> Dict[str, Any]:
        """Add product to user's shopping cart."""
        try:
            # Check if product exists
            stmt = select(Product).where(Product.product_id == cart_data.product_id)
            result = await db.execute(stmt)
            product = result.scalar_one_or_none()

            if not product:
                raise ValueError(f"Product {cart_data.product_id} not found")
            
            # Check if item already in cart
            stmt = select(CartItem).where(
                CartItem.user_id == user_id,
                CartItem.product_id == cart_data.product_id
            )
            result = await db.execute(stmt)
            existing_item = result.scalar_one_or_none()
            
            if existing_item:
                # Update quantity
                existing_item.quantity += cart_data.quantity
                cart_item = existing_item
            else:
                # Create new cart item
                cart_item = CartItem(
                    user_id=user_id,
                    product_id=cart_data.product_id,
                    quantity=cart_data.quantity,
                    added_at=datetime.now(timezone.utc)
                )
                db.add(cart_item)
            
            await db.flush()
            
            # Log the action
            user_action = UserAction(
                user_id=user_id,
                product_id=cart_data.product_id,
                action_type='add_to_cart',
                reward=2.0,  # Higher reward for cart actions
                session_time=0,
                action_timestamp=datetime.now(timezone.utc)
            )
            db.add(user_action)
            
            # Update learning system
            if learning_manager.is_ready:
                await learning_manager.update_user_preferences(db, user_id, cart_data.product_id, 'add_to_cart')
            
            await db.commit()
            
            return {
                'success': True,
                'cart_item_id': cart_item.cart_item_id,
                'message': 'Product added to cart successfully'
            }
            
        except Exception as e:
            await db.rollback()
            raise ValueError(f"Failed to add product to cart: {str(e)}")
    
    async def get_cart(self, db: AsyncSession, user_id: int) -> CartResponse:
        """Get user's shopping cart with product details."""
        try:
            stmt = select(CartItem).where(CartItem.user_id == user_id)
            result = await db.execute(stmt)
            cart_items_db = result.scalars().all()
            
            cart_items = []
            total_price = 0.0
            total_quantity = 0
            
            for cart_item in cart_items_db:
                # Get product details for each cart item
                product_stmt = select(Product).where(Product.product_id == cart_item.product_id)
                product_result = await db.execute(product_stmt)
                product = product_result.scalar_one_or_none()
                
                if product:
                    item_total = product.price * cart_item.quantity
                    total_price += item_total
                    total_quantity += cart_item.quantity
                    
                    cart_items.append({
                        'cart_item_id': cart_item.cart_item_id,
                        'product_id': cart_item.product_id,
                        'quantity': cart_item.quantity,
                        'added_at': cart_item.added_at,
                        'product_info': ProductInfo(
                            product_id=product.product_id,
                            product_name=product.product_name,
                            name_format=product.name_format,
                            category_name=product.category_name,
                            price=product.price,
                            popularity=product.popularity,
                            quality=product.quality
                        ),
                        'item_total': item_total
                    })
            
            return CartResponse(
                user_id=user_id,
                cart_items=cart_items,
                total_items=len(cart_items),
                total_quantity=total_quantity,
                total_price=total_price
            )
            
        except Exception as e:
            raise ValueError(f"Failed to get cart: {str(e)}")
    
    async def update_cart_item(self, db: AsyncSession, user_id: int, cart_item_id: int, quantity: int) -> Dict[str, Any]:
        """Update cart item quantity."""
        try:
            if quantity < 1:
                raise ValueError("Quantity must be at least 1")
            
            stmt = select(CartItem).where(
                CartItem.cart_item_id == cart_item_id,
                CartItem.user_id == user_id
            )
            result = await db.execute(stmt)
            cart_item = result.scalar_one_or_none()
            
            if not cart_item:
                raise ValueError("Cart item not found")
            
            cart_item.quantity = quantity
            await db.commit()
            
            return {
                'success': True,
                'message': 'Cart item updated successfully'
            }
            
        except Exception as e:
            await db.rollback()
            raise ValueError(f"Failed to update cart item: {str(e)}")
    
    async def clear_cart(self, db: AsyncSession, user_id: int) -> Dict[str, Any]:
        """Clear user's entire cart."""
        try:
            stmt = select(CartItem).where(CartItem.user_id == user_id)
            result = await db.execute(stmt)
            cart_items = result.scalars().all()
            
            for cart_item in cart_items:
                await db.delete(cart_item)
            
            await db.commit()
            
            return {
                'success': True,
                'message': 'Cart cleared successfully'
            }
            
        except Exception as e:
            await db.rollback()
            raise ValueError(f"Failed to clear cart: {str(e)}")
    
    async def place_order(self, db: AsyncSession, user_id: int) -> OrderResponse:
        """Place an order from the shopping cart."""
        try:
            # Get cart items
            stmt = select(CartItem).where(CartItem.user_id == user_id)
            result = await db.execute(stmt)
            cart_items = result.scalars().all()
            
            if not cart_items:
                raise ValueError("Cart is empty")
            
            # Calculate total price and prepare items JSON
            total_price = 0.0
            items_data = []
            
            for cart_item in cart_items:
                product_stmt = select(Product).where(Product.product_id == cart_item.product_id)
                product_result = await db.execute(product_stmt)
                product = product_result.scalar_one_or_none()
                
                if product:
                    item_total = product.price * cart_item.quantity
                    total_price += item_total
                    
                    items_data.append({
                        'product_id': product.product_id,
                        'category': product.category_name,
                        'quantity': cart_item.quantity,
                        'price': product.price,
                        'item_total': item_total
                    })
            
            # Create order
            order = Order(
                user_id=user_id,
                total_price=total_price,
                status='confirmed',
                order_date=datetime.now(timezone.utc),
                items=items_data
            )
            
            db.add(order)
            await db.flush()
            
            # Log purchase actions for each item
            for cart_item in cart_items:
                user_action = UserAction(
                    user_id=user_id,
                    product_id=cart_item.product_id,
                    action_type='purchase',
                    reward=5.0,  # High reward for purchases
                    session_time=0,
                    action_timestamp=datetime.now(timezone.utc)
                )
                db.add(user_action)
                
                # Update learning system
                if learning_manager.is_ready:
                    await learning_manager.update_user_preferences(db, user_id, cart_item.product_id, 'purchase')
            
            # Clear cart after order
            for cart_item in cart_items:
                await db.delete(cart_item)
            
            await db.commit()
            
            return OrderResponse(
                message='Order placed successfully!',
                order={
                    'order_id': order.order_id,
                    'user_id': order.user_id,
                    'total_price': order.total_price,
                    'status': order.status,
                    'order_date': order.order_date,
                    'items': items_data
                }
            )
            
        except Exception as e:
            await db.rollback()
            raise ValueError(f"Failed to place order: {str(e)}")
    
    async def get_order_history(self, db: AsyncSession, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's order history."""
        try:
            stmt = select(Order).where(
                Order.user_id == user_id
            ).order_by(Order.order_date.desc()).limit(limit)
            result = await db.execute(stmt)
            orders = result.scalars().all()
            
            order_history = []
            for order in orders:
                order_history.append({
                    'order_id': order.order_id,
                    'total_price': order.total_price,
                    'status': order.status,
                    'order_date': order.order_date,
                    'completed_at': order.completed_at,
                    'items_count': len(order.items) if order.items else 0
                })
            
            return order_history
            
        except Exception as e:
            raise ValueError(f"Failed to get order history: {str(e)}")
    
    async def remove_from_cart(self, db: AsyncSession, user_id: int, cart_item_id: int) -> Dict[str, Any]:
        """Remove item from cart."""
        try:
            stmt = select(CartItem).where(
                CartItem.cart_item_id == cart_item_id,
                CartItem.user_id == user_id
            )
            result = await db.execute(stmt)
            cart_item = result.scalar_one_or_none()
            
            if not cart_item:
                raise ValueError("Cart item not found")
            
            product_id = cart_item.product_id
                        
            # Log the removal action with reward
            user_action = UserAction(
                user_id=user_id,
                product_id=product_id,
                action_type='remove_from_cart',
                reward=-0.5,
                session_time=0,
                action_timestamp=datetime.now(timezone.utc)
            )
            db.add(user_action)
            
            await db.delete(cart_item)
            await db.commit()
            
            return {
                'success': True,
                'message': 'Item removed from cart successfully',
                'reward': -0.5
            }
            
        except Exception as e:
            await db.rollback()
            raise ValueError(f"Failed to remove item from cart: {str(e)}")
