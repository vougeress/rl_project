import { useState, useEffect, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Product } from '../../types';
import { apiService } from '../../services/api';
import { ProductCard } from '../product/ProductCard';
import { ShoppingCart, LogOut } from 'lucide-react';
import './Home.css';

interface HomeProps {
  userId: number;
  userName: string;
  onLogout: () => Promise<void>;
}

export function Home({ userId, userName, onLogout }: HomeProps) {
  const navigate = useNavigate();
  const [products, setProducts] = useState<Product[]>([]);
  const [filteredProducts, setFilteredProducts] = useState<Product[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [cartCount, setCartCount] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const recommendationCacheKey = useMemo(
    () => `user_${userId}_recommendations`,
    [userId]
  );

  useEffect(() => {
    if (selectedCategory === 'all') {
      setFilteredProducts(products);
    } else {
      setFilteredProducts(products.filter(p => p.category_name === selectedCategory));
    }
  }, [selectedCategory, products]);

  const loadProducts = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await apiService.getRecommendations(userId, 20);
      setProducts(response.products);
      setFilteredProducts(response.products);
      sessionStorage.setItem(recommendationCacheKey, JSON.stringify(response.products));
    } catch (error) {
      console.error('Failed to load products:', error);
      setError(error instanceof Error ? error.message : 'Failed to load products');
    } finally {
      setIsLoading(false);
    }
  }, [userId, recommendationCacheKey]);

  const loadCartCount = useCallback(async () => {
    try {
      const cart = await apiService.getCart(userId);
      setCartCount(cart.total_quantity);
    } catch (error) {
      console.error('Failed to load cart:', error);
      setError(error instanceof Error ? error.message : 'Failed to load cart info');
      setCartCount(0);
    }
  }, [userId]);

  useEffect(() => {
    const cached = sessionStorage.getItem(recommendationCacheKey);
    if (cached) {
      try {
        const parsed = JSON.parse(cached) as Product[];
        setProducts(parsed);
        setFilteredProducts(parsed);
        setIsLoading(false);
      } catch (err) {
        console.error('Failed to parse cached recommendations:', err);
        sessionStorage.removeItem(recommendationCacheKey);
        loadProducts();
      }
    } else {
      loadProducts();
    }

    loadCartCount();
  }, [userId, recommendationCacheKey, loadProducts, loadCartCount]);

  const handleProductClick = (product: Product) => {
    // Placeholder for future telemetry
    console.debug('Navigating to product', product.product_id);
    navigate(`/product/${product.product_id}`);
  };

  const handleCartClick = () => {
    navigate('/cart');
  };

  const handleLogoutClick = async () => {
    try {
      await onLogout();
    } catch (err) {
      console.error('Logout failed:', err);
    }
  };

  return (
    <div className="home-container">
      <header className="home-header">
        <div className="header-content">
          <h1 className="header-title">ShopSmart</h1>
          
          <div className="header-actions">
            <span className="user-info">Welcome, {userName}!</span>
            
            <button className="cart-button" onClick={handleCartClick}>
              <ShoppingCart size={20} />
              <span>Cart</span>
              {cartCount > 0 && <span className="cart-badge">{cartCount}</span>}
            </button>
            
            <button className="logout-button" onClick={handleLogoutClick}>
              <LogOut size={18} />
            </button>
          </div>
        </div>
      </header>

      <main className="home-content">
        {error && (
          <div className="error-banner">
            {error}
          </div>
        )}

        {isLoading ? (
          <div className="loading-container">
            <div className="loading-spinner"></div>
          </div>
        ) : filteredProducts.length > 0 ? (
          <div className="products-grid">
            {filteredProducts.map(product => (
              <ProductCard
                key={product.product_id}
                product={product}
                onClick={() => handleProductClick(product)}
              />
            ))}
          </div>
        ) : (
          <div className="empty-state">
            <p>No products found in this category.</p>
          </div>
        )}
      </main>
    </div>
  );
}
