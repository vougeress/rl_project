"""
Shopping cart service for handling cart operations and orders.
"""

import uuid
from typing import Dict, List, Any
from datetime import datetime

from api.core.learning_manager import learning_manager
from api.models.schemas import UserAction, ProductActionResponse, CartResponse, OrderResponse


class CartService:
    def __init__(self):
        self.shopping_carts = {}
    
    def handle_product_action(self, action_data: UserAction) -> ProductActionResponse:
        """Handle user actions on products with continuous learning."""
        # Validate action type
        valid_actions = ['like', 'dislike', 'report', 'add_to_cart']
        if action_data.action not in valid_actions:
            raise ValueError(f"Invalid action. Must be one of: {valid_actions}")
        
        # Ensure learning system is ready
        if not learning_manager.is_ready:
            learning_manager.initialize_system()
        
        # Validate product
        if action_data.product_id >= learning_manager.catalog.n_products:
            raise ValueError(f"Product ID must be less than {learning_manager.catalog.n_products}")
        
        # Get product info
        product_info = learning_manager.catalog.get_product_info(action_data.product_id)
        
        # Handle different actions
        response_messages = {
            'like': 'Товар добавлен в избранное',
            'dislike': 'Отмечено как неинтересное',
            'report': 'Жалоба отправлена',
            'add_to_cart': 'Товар добавлен в корзину'
        }
        
        # For add_to_cart, actually add to cart
        if action_data.action == 'add_to_cart':
            if action_data.user_id not in self.shopping_carts:
                self.shopping_carts[action_data.user_id] = []
            
            # Check if product already in cart
            existing_item = None
            for item in self.shopping_carts[action_data.user_id]:
                if item['product_id'] == action_data.product_id:
                    existing_item = item
                    break
            
            if existing_item:
                existing_item['quantity'] += 1
            else:
                self.shopping_carts[action_data.user_id].append({
                    'product_id': action_data.product_id,
                    'quantity': 1,
                    'added_at': datetime.now()
                })
        
        # Calculate reward based on action
        rewards = {
            'like': 0.5,
            'dislike': -0.2,
            'report': -1.0,
            'add_to_cart': 2.0
        }
        
        reward = rewards[action_data.action]
        
        # CONTINUOUS LEARNING: Update agent based on user action
        learning_manager.learn_from_action(
            action_data.user_id,
            action_data.product_id,
            action_data.action,
            reward
        )
        
        from api.models.schemas import ProductInfo
        return ProductActionResponse(
            user_id=action_data.user_id,
            product_id=action_data.product_id,
            product_info=ProductInfo(**product_info),
            action=action_data.action,
            message=response_messages[action_data.action],
            reward=reward,
            timestamp=datetime.now()
        )
    
    def get_shopping_cart(self, user_id: int) -> CartResponse:
        """Get user's shopping cart."""
        if user_id not in self.shopping_carts:
            self.shopping_carts[user_id] = []
        
        # Ensure learning system is ready
        if not learning_manager.is_ready:
            learning_manager.initialize_system()
        
        # Enrich cart items with product info
        cart_items = []
        total_price = 0
        
        for item in self.shopping_carts[user_id]:
            product_info = learning_manager.catalog.get_product_info(item['product_id'])
            item_total = product_info['price'] * item['quantity']
            total_price += item_total
            
            cart_items.append({
                'product_id': item['product_id'],
                'quantity': item['quantity'],
                'product_info': product_info,
                'item_total': item_total,
                'added_at': item.get('added_at')
            })
        
        return CartResponse(
            user_id=user_id,
            cart_items=cart_items,
            total_items=len(cart_items),
            total_quantity=sum(item['quantity'] for item in self.shopping_carts[user_id]),
            total_price=total_price
        )
    
    def place_order(self, user_id: int) -> OrderResponse:
        """Place an order from the shopping cart."""
        if user_id not in self.shopping_carts or not self.shopping_carts[user_id]:
            raise ValueError("Корзина пуста")
        
        # Get cart info
        cart_response = self.get_shopping_cart(user_id)
        
        # Create order
        order_id = str(uuid.uuid4())
        order = {
            'order_id': order_id,
            'user_id': user_id,
            'items': cart_response.cart_items,
            'total_price': cart_response.total_price,
            'order_date': datetime.now(),
            'status': 'confirmed'
        }
        
        # Clear the cart
        self.shopping_carts[user_id] = []
        
        return OrderResponse(
            message='Спасибо за заказ!',
            order=order
        )