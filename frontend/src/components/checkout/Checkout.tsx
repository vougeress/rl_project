import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { CartResponse } from '../../types';
import { apiService } from '../../services/api';
import { ArrowLeft, CheckCircle } from 'lucide-react';
import './Checkout.css';

interface CheckoutProps {
  userId: number;
  userName: string;
}

export function Checkout({ userId, userName }: CheckoutProps) {
  const navigate = useNavigate();
  const [cart, setCart] = useState<CartResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);
  const [orderId, setOrderId] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  const [formData, setFormData] = useState({
    fullName: userName || '',
    email: '',
    phone: '',
    address: '',
    city: '',
    zipCode: '',
    notes: ''
  });

  useEffect(() => {
    loadCart();
  }, [userId]);

  const loadCart = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const cartData = await apiService.getCart(userId);
      setCart(cartData);
      
      if (!cartData || cartData.cart_items.length === 0) {
        navigate('/cart');
      }
    } catch (error) {
      console.error('Failed to load cart:', error);
      setError(error instanceof Error ? error.message : 'Failed to load cart');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);

    try {
      const response = await apiService.placeOrder(userId);
      setOrderId(response.order?.order_id || `ORD-${Date.now()}`);
      setShowSuccess(true);
    } catch (error) {
      console.error('Failed to place order:', error);
      setError(error instanceof Error ? error.message : 'Failed to place order. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleSuccessClose = () => {
    setShowSuccess(false);
    navigate('/');
  };

  if (isLoading) {
    return (
      <div className="checkout-container">
        <div className="loading-detail">
          <div className="loading-spinner"></div>
        </div>
      </div>
    );
  }

  if (!cart || cart.cart_items.length === 0) {
    if (error) {
      return (
        <div className="checkout-container">
          <div className="error-banner">{error}</div>
        </div>
      );
    }
    return null;
  }

  return (
    <div className="checkout-container">
      <header className="checkout-header">
        <div className="checkout-header-content">
          <button className="back-button" onClick={() => navigate('/cart')}>
            <ArrowLeft size={20} />
            <span>Back to Cart</span>
          </button>
          
          <h1 className="header-title">Checkout</h1>
        </div>
      </header>

      <main className="checkout-content">
        {error && (
          <div className="error-banner">
            {error}
          </div>
        )}
        <div className="checkout-card">
          <h2 className="checkout-title">Complete Your Order</h2>

          <form onSubmit={handleSubmit} className="checkout-form">
            <div className="form-section">
              <h3 className="section-title">Contact Information</h3>

              <div className="form-group">
                <label htmlFor="fullName" className="form-label">Full Name</label>
                <input
                  id="fullName"
                  type="text"
                  className="form-input"
                  value={formData.fullName}
                  onChange={(e) => setFormData({ ...formData, fullName: e.target.value })}
                  required
                />
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="email" className="form-label">Email</label>
                  <input
                    id="email"
                    type="email"
                    className="form-input"
                    value={formData.email}
                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                    required
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="phone" className="form-label">Phone</label>
                  <input
                    id="phone"
                    type="tel"
                    className="form-input"
                    value={formData.phone}
                    onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
                    required
                  />
                </div>
              </div>
            </div>

            <div className="form-section">
              <h3 className="section-title">Delivery Address</h3>

              <div className="form-group">
                <label htmlFor="address" className="form-label">Street Address</label>
                <input
                  id="address"
                  type="text"
                  className="form-input"
                  value={formData.address}
                  onChange={(e) => setFormData({ ...formData, address: e.target.value })}
                  required
                />
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="city" className="form-label">City</label>
                  <input
                    id="city"
                    type="text"
                    className="form-input"
                    value={formData.city}
                    onChange={(e) => setFormData({ ...formData, city: e.target.value })}
                    required
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="zipCode" className="form-label">ZIP Code</label>
                  <input
                    id="zipCode"
                    type="text"
                    className="form-input"
                    value={formData.zipCode}
                    onChange={(e) => setFormData({ ...formData, zipCode: e.target.value })}
                    required
                  />
                </div>
              </div>
            </div>

            <div className="form-section">
              <h3 className="section-title">Additional Notes</h3>
              <div className="form-group">
                <textarea
                  id="notes"
                  className="form-input"
                  rows={3}
                  value={formData.notes}
                  onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
                  placeholder="Any special instructions?"
                  style={{ resize: 'vertical', fontFamily: 'inherit' }}
                />
              </div>
            </div>

            <div className="form-section">
              <h3 className="section-title">Order Summary</h3>
              
              <div className="order-summary-items">
                {cart.cart_items.map((item) => (
                  <div key={item.cart_item_id} className="order-item">
                    <div className="item-details">
                      <div className="item-name">{item.product.product_name}</div>
                      <div>Quantity: {item.quantity}</div>
                    </div>
                    <div>${item.item_total.toFixed(2)}</div>
                  </div>
                ))}
              </div>

              <div className="order-total">
                <span>Total Amount</span>
                <span>${cart.total_price.toFixed(2)}</span>
              </div>
            </div>

            <div className="checkout-actions">
              <button
                type="button"
                className="cancel-button"
                onClick={() => navigate('/cart')}
              >
                Cancel
              </button>
              
              <button
                type="submit"
                className="submit-order-button"
                disabled={isSubmitting}
              >
                {isSubmitting ? 'Processing...' : 'Place Order'}
              </button>
            </div>
          </form>
        </div>
      </main>

      {showSuccess && (
        <div className="success-modal" onClick={handleSuccessClose}>
          <div className="success-modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="success-icon">
              <CheckCircle size={48} />
            </div>
            <h2 className="success-modal-title">Order Placed Successfully!</h2>
            <p className="success-modal-text">
              Your order #{orderId} has been confirmed.
              <br />
              We'll send you a confirmation email shortly.
            </p>
            <button className="success-modal-button" onClick={handleSuccessClose}>
              Continue Shopping
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
