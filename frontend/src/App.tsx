import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Login } from './components/auth/Login';
import { Home } from './components/home/Home';
import { ProductDetail } from './components/product/ProductDetail';
import { CartPage } from './components/cart/CartPage';
import { Checkout } from './components/checkout/Checkout';
import { apiService } from './services/api';
import { UserRegistrationResponse } from './types';

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

  const handleLogout = async () => {
    const storedUser = localStorage.getItem('currentUser');
    let parsedUser: UserRegistrationResponse | null = null;

    if (storedUser) {
      try {
        parsedUser = JSON.parse(storedUser) as UserRegistrationResponse;
      } catch (error) {
        console.error('Failed to parse stored user during logout:', error);
      }
    }

    if (parsedUser?.session_id) {
      try {
        await apiService.endSession(parsedUser.user_id, parsedUser.session_id);
      } catch (error) {
        console.error('Failed to end user session:', error);
      }
    }

    const cacheUserId = parsedUser?.user_id ?? userId;
    if (cacheUserId) {
      sessionStorage.removeItem(`user_${cacheUserId}_recommendations`);
    }

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
