import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Product } from '../../types';
import { apiService } from '../../services/api';
import { ProductCard } from '../product/ProductCard';
import { ShoppingCart, LogOut } from 'lucide-react';
import './Home.css';

interface HomeProps {
  userId: number;
  userName: string;
  onLogout: () => void;
}

export function Home({ userId, userName, onLogout }: HomeProps) {
  const navigate = useNavigate();
  const [products, setProducts] = useState<Product[]>([]);
  const [filteredProducts, setFilteredProducts] = useState<Product[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [cartCount, setCartCount] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadProducts();
    loadCartCount();
  }, [userId]);

  useEffect(() => {
    if (selectedCategory === 'all') {
      setFilteredProducts(products);
    } else {
      setFilteredProducts(products.filter(p => p.category_name === selectedCategory));
    }
  }, [selectedCategory, products]);

  const loadProducts = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await apiService.getRecommendations(userId, 20);
      setProducts(response.products);
      setFilteredProducts(response.products);
    } catch (error) {
      console.error('Failed to load products:', error);
      setError(error instanceof Error ? error.message : 'Failed to load products');
    } finally {
      setIsLoading(false);
    }
  };

  const loadCartCount = async () => {
    try {
      const cart = await apiService.getCart(userId);
      setCartCount(cart.total_quantity);
    } catch (error) {
      console.error('Failed to load cart:', error);
      setError(error instanceof Error ? error.message : 'Failed to load cart info');
      setCartCount(0);
    }
  };

  const handleProductClick = async (product: Product) => {
    // Track view action
    try {
      await apiService.processAction(userId, {
        user_id: userId,
        product_id: product.product_id,
        action_type: 'view'
      });
    } catch (err) {
      console.error('Failed to log view action:', err);
      setError(err instanceof Error ? err.message : 'Failed to log action');
    } finally {
      navigate(`/product/${product.product_id}`);
    }
  };

  const handleCartClick = () => {
    navigate('/cart');
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
            
            <button className="logout-button" onClick={onLogout}>
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
