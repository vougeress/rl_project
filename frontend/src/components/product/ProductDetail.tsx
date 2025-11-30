import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Product } from '../../types';
import { apiService } from '../../services/api';
import { 
  ArrowLeft, 
  ShoppingCart, 
  ThumbsUp, 
  ThumbsDown, 
  Flag,
  Star,
  TrendingUp
} from 'lucide-react';
import { ImageWithFallback } from '../figma/ImageWithFallback';
import './ProductDetail.css';

interface ProductDetailProps {
  userId: number;
}

export function ProductDetail({ userId }: ProductDetailProps) {
  const { productId } = useParams<{ productId: string }>();
  const navigate = useNavigate();
  const [product, setProduct] = useState<Product | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [liked, setLiked] = useState(false);
  const [disliked, setDisliked] = useState(false);
  const [reported, setReported] = useState(false);
  const [addedToCart, setAddedToCart] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (productId) {
      loadProduct(parseInt(productId));
    }
  }, [productId]);

  const loadProduct = async (id: number) => {
    setIsLoading(true);
    setError(null);
    try {
      const productData = await apiService.getProductInfo(id);
      setProduct(productData);
    } catch (error) {
      console.error('Failed to load product:', error);
      setError(error instanceof Error ? error.message : 'Failed to load product');
    } finally {
      setIsLoading(false);
    }
  };

  const handleLike = async () => {
    if (!product) return;
    
    const newLiked = !liked;
    setLiked(newLiked);
    if (newLiked) setDisliked(false);
    try {
      setError(null);
      await apiService.processAction(userId, {
        user_id: userId,
        product_id: product.product_id,
        action_type: 'like'
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send like');
      setLiked(!newLiked);
    }
  };

  const handleDislike = async () => {
    if (!product) return;
    
    const newDisliked = !disliked;
    setDisliked(newDisliked);
    if (newDisliked) setLiked(false);
    try {
      setError(null);
      await apiService.processAction(userId, {
        user_id: userId,
        product_id: product.product_id,
        action_type: 'dislike'
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send dislike');
      setDisliked(!newDisliked);
    }
  };

  const handleReport = async () => {
    if (!product) return;
    
    const newReported = !reported;
    setReported(newReported);
    try {
      setError(null);
      await apiService.processAction(userId, {
        user_id: userId,
        product_id: product.product_id,
        action_type: newReported ? 'report' : 'view'
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send report');
      setReported(!newReported);
    }
  };

  const handleAddToCart = async () => {
    if (!product) return;

    try {
      setError(null);
      await apiService.addToCart(userId, product.product_id, 1);
      
      await apiService.processAction(userId, {
        user_id: userId,
        product_id: product.product_id,
        action_type: 'add_to_cart'
      });

      setAddedToCart(true);
      setTimeout(() => setAddedToCart(false), 3000);
    } catch (error) {
      console.error('Failed to add to cart:', error);
      setError(error instanceof Error ? error.message : 'Failed to add to cart');
    }
  };

  const handleGoToCart = () => {
    navigate('/cart');
  };

  if (isLoading) {
    return (
      <div className="product-detail-container">
        <div className="loading-detail">
          <div className="loading-spinner"></div>
        </div>
      </div>
    );
  }

  if (!product) {
    return (
      <div className="product-detail-container">
        <div className="empty-state">
          <p>{error ?? 'Product not found'}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="product-detail-container">
      <header className="product-detail-header">
        <div className="detail-header-content">
          <button className="back-button" onClick={() => navigate('/')}>
            <ArrowLeft size={20} />
            <span>Back to Products</span>
          </button>

          <button className="cart-button" onClick={handleGoToCart}>
            <ShoppingCart size={20} />
            <span>View Cart</span>
          </button>
        </div>
      </header>

      <main className="detail-content">
        <div className="product-detail-grid">
          <div className="product-detail-image-wrapper">
            <ImageWithFallback
              src={product.image_url || ''}
              alt={product.product_name}
              className="product-detail-image"
            />
          </div>

          <div className="product-detail-info">
            <div className="detail-category">{product.category_name}</div>
            <h1 className="detail-title">{product.product_name}</h1>
            <div className="detail-price">${product.price.toFixed(2)}</div>

            <div className="detail-stats">
              <div className="stat-item">
                <span className="stat-label">Quality</span>
                <div className="stat-value">
                  <Star className="stat-stars" size={20} fill="currentColor" />
                  <span>{product.quality.toFixed(1)}/1.0</span>
                </div>
              </div>

              <div className="stat-item">
                <span className="stat-label">Popularity</span>
                <div className="stat-value">
                  <TrendingUp size={20} />
                  <span>{Math.round(product.popularity * 100)}%</span>
                </div>
              </div>
            </div>

            {addedToCart && (
              <div className="success-message">
                ✓ Added to cart successfully!
              </div>
            )}

            {error && (
              <div className="error-message" style={{ marginTop: '0.75rem' }}>
                {error}
              </div>
            )}

            <div className="action-buttons">
              <button className="primary-button" onClick={handleAddToCart}>
                <ShoppingCart size={20} />
                <span>Add to Cart</span>
              </button>

              <div className="secondary-actions">
                <button 
                  className={`icon-button ${liked ? 'liked' : ''}`}
                  onClick={handleLike}
                >
                  <ThumbsUp size={20} />
                  <span>Like</span>
                </button>

                <button 
                  className={`icon-button ${disliked ? 'disliked' : ''}`}
                  onClick={handleDislike}
                >
                  <ThumbsDown size={20} />
                  <span>Dislike</span>
                </button>

                <button 
                  className={`icon-button ${reported ? 'reported' : ''}`}
                  onClick={handleReport}
                >
                  <Flag size={20} />
                  <span>Report</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
