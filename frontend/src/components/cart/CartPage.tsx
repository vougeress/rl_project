import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { CartResponse } from '../../types';
import { apiService } from '../../services/api';
import { ArrowLeft, ShoppingCart, Trash2, Plus, Minus } from 'lucide-react';
import { ImageWithFallback } from '../figma/ImageWithFallback';
import './CartPage.css';

interface CartPageProps {
  userId: number;
}

export function CartPage({ userId }: CartPageProps) {
  const navigate = useNavigate();
  const [cart, setCart] = useState<CartResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadCart();
  }, [userId]);

  const loadCart = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const cartData = await apiService.getCart(userId);
      setCart(cartData);
    } catch (error) {
      console.error('Failed to load cart:', error);
      setError(error instanceof Error ? error.message : 'Failed to load cart');
    } finally {
      setIsLoading(false);
    }
  };

  const handleUpdateQuantity = async (cartItemId: number, newQuantity: number) => {
    if (newQuantity < 1) return;
    
    try {
      setError(null);
      await apiService.updateCartItem(userId, cartItemId, newQuantity);
      await loadCart();
    } catch (error) {
      console.error('Failed to update quantity:', error);
      setError(error instanceof Error ? error.message : 'Failed to update quantity');
    }
  };

  const handleRemoveItem = async (cartItemId: number) => {
    try {
      setError(null);
      await apiService.removeFromCart(userId, cartItemId);
      await loadCart();
    } catch (error) {
      console.error('Failed to remove item:', error);
      setError(error instanceof Error ? error.message : 'Failed to remove item');
    }
  };

  const handleClearCart = async () => {
    if (!window.confirm('Are you sure you want to clear your cart?')) return;
    
    try {
      setError(null);
      await apiService.clearCart(userId);
      await loadCart();
    } catch (error) {
      console.error('Failed to clear cart:', error);
      setError(error instanceof Error ? error.message : 'Failed to clear cart');
    }
  };

  const handleCheckout = () => {
    navigate('/checkout');
  };

  if (isLoading) {
    return (
      <div className="cart-page-container">
        <div className="loading-detail">
          <div className="loading-spinner"></div>
        </div>
      </div>
    );
  }

  const isEmpty = !cart || cart.cart_items.length === 0;

  return (
    <div className="cart-page-container">
      <header className="cart-page-header">
        <div className="cart-header-content">
          <button className="back-button" onClick={() => navigate('/')}>
            <ArrowLeft size={20} />
            <span>Continue Shopping</span>
          </button>
          
          <h1 className="header-title">Shopping Cart</h1>
        </div>
      </header>

      <main className="cart-page-content">
        {error && (
          <div className="error-banner">
            {error}
          </div>
        )}
        {isEmpty ? (
          <div className="empty-cart">
            <ShoppingCart size={64} className="empty-cart-icon" />
            <h2 className="empty-cart-text">Your cart is empty</h2>
            <button 
              className="continue-shopping-button"
              onClick={() => navigate('/')}
            >
              Start Shopping
            </button>
          </div>
        ) : (
          <div className="cart-layout">
            <div className="cart-items-section">
              {cart.cart_items.map((item) => (
                <div key={item.cart_item_id} className="cart-item-card">
                  <div className="cart-item-image-wrapper">
                    <ImageWithFallback
                      src={item.product.image_url || ''}
                      alt={item.product.product_name}
                      className="cart-item-image"
                    />
                  </div>

                  <div className="cart-item-info">
                    <div className="cart-item-category">
                      {item.product.category_name}
                    </div>
                    <h3 className="cart-item-name">{item.product.product_name}</h3>
                    <div className="cart-item-price">
                      ${item.product.price.toFixed(2)} each
                    </div>

                    <div className="cart-item-actions">
                      <div className="quantity-control">
                        <button
                          className="quantity-button"
                          onClick={() => handleUpdateQuantity(item.cart_item_id, item.quantity - 1)}
                          disabled={item.quantity <= 1}
                        >
                          <Minus size={16} />
                        </button>
                        <span className="quantity-value">{item.quantity}</span>
                        <button
                          className="quantity-button"
                          onClick={() => handleUpdateQuantity(item.cart_item_id, item.quantity + 1)}
                        >
                          <Plus size={16} />
                        </button>
                      </div>

                      <div className="cart-item-price">
                        Total: ${item.item_total.toFixed(2)}
                      </div>

                      <button
                        className="remove-button"
                        onClick={() => handleRemoveItem(item.cart_item_id)}
                      >
                        <Trash2 size={16} />
                        <span>Remove</span>
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="cart-summary-section">
              <div className="cart-summary">
                <h3 className="summary-title">Order Summary</h3>

                <div className="summary-row">
                  <span>Items ({cart.total_items})</span>
                  <span>${cart.total_price.toFixed(2)}</span>
                </div>

                <div className="summary-row">
                  <span>Total Quantity</span>
                  <span>{cart.total_quantity}</span>
                </div>

                <div className="summary-row total">
                  <span>Total</span>
                  <span>${cart.total_price.toFixed(2)}</span>
                </div>

                <button className="checkout-button" onClick={handleCheckout}>
                  Proceed to Checkout
                </button>

                <button className="clear-cart-button" onClick={handleClearCart}>
                  Clear Cart
                </button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
