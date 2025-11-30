import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Login } from './components/auth/Login';
import { Home } from './components/home/Home';
import { ProductDetail } from './components/product/ProductDetail';
import { CartPage } from './components/cart/CartPage';
import { Checkout } from './components/checkout/Checkout';

function App() {
  const [userId, setUserId] = useState<number | null>(null);
  const [userName, setUserName] = useState<string>('');

  useEffect(() => {
    // Check if user is already logged in
    const storedUser = localStorage.getItem('currentUser');
    if (storedUser) {
      try {
        const user = JSON.parse(storedUser);
        setUserId(user.user_id);
        setUserName(user.name);
      } catch (error) {
        console.error('Failed to parse stored user:', error);
        localStorage.removeItem('currentUser');
      }
    }
  }, []);

  const handleLogin = (newUserId: number) => {
    setUserId(newUserId);
    
    const storedUser = localStorage.getItem('currentUser');
    if (storedUser) {
      const user = JSON.parse(storedUser);
      setUserName(user.name);
    }
  };

  const handleLogout = () => {
    setUserId(null);
    setUserName('');
    localStorage.removeItem('currentUser');
  };

  return (
    <BrowserRouter>
      <Routes>
        {!userId ? (
          <>
            <Route path="/" element={<Login onLogin={handleLogin} />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </>
        ) : (
          <>
            <Route 
              path="/" 
              element={
                <Home 
                  userId={userId} 
                  userName={userName} 
                  onLogout={handleLogout} 
                />
              } 
            />
            <Route 
              path="/product/:productId" 
              element={<ProductDetail userId={userId} />} 
            />
            <Route 
              path="/cart" 
              element={<CartPage userId={userId} />} 
            />
            <Route 
              path="/checkout" 
              element={<Checkout userId={userId} userName={userName} />} 
            />
            <Route path="*" element={<Navigate to="/" replace />} />
          </>
        )}
      </Routes>
    </BrowserRouter>
  );
}

export default App;
