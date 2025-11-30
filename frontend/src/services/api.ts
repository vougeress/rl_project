import {
  User,
  UserRegistration,
  UserRegistrationResponse,
  Product,
  RecommendationsResponse,
  UserAction,
  CartResponse,
  CartItem,
  CategoryName,
  CATEGORIES
} from '../types';
import { getRandomCategoryImage } from '../utils/categoryImages';

const API_BASE_URL = import.meta.env?.VITE_API_URL ?? 'http://localhost:8000';

type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE';

type ApiCartItem = {
  cart_item_id: number;
  product_id: number;
  quantity: number;
  item_total?: number;
  added_at?: string;
  product?: Partial<Product> & Record<string, any>;
  product_info?: Partial<Product> & Record<string, any>;
};

type ApiCartResponse = {
  user_id: number;
  cart_items: ApiCartItem[];
  total_items: number;
  total_quantity: number;
  total_price: number;
};

const CATEGORY_FALLBACK: CategoryName = 'Electronics';

const resolveCategoryForImage = (categoryName?: string): CategoryName => {
  if (!categoryName) {
    return CATEGORY_FALLBACK;
  }
  return (CATEGORIES.includes(categoryName as CategoryName)
    ? categoryName
    : CATEGORY_FALLBACK) as CategoryName;
};

const coerceNumber = (value: unknown, fallback = 0): number => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const normalizeProduct = (raw: Partial<Product> & Record<string, any>): Product => {
  const productId = raw.product_id ?? raw.id ?? 0;
  const categoryName = raw.category_name ?? raw.category ?? 'General';
  const normalized: Product = {
    product_id: productId,
    product_name: raw.product_name ?? raw.name ?? `Product #${productId}`,
    category_name: categoryName,
    price: coerceNumber(raw.price),
    popularity: coerceNumber(raw.popularity),
    quality: coerceNumber(raw.quality),
    name_format: raw.name_format ?? null,
    category_id: raw.category_id,
    style_vector: raw.style_vector ?? raw.styleVector,
    image_url: raw.image_url ?? getRandomCategoryImage(resolveCategoryForImage(categoryName))
  };
  return normalized;
};

const parseErrorMessage = async (response: Response): Promise<string> => {
  try {
    const data = await response.json();
    if (typeof data === 'string') {
      return data;
    }
    if (data?.detail) {
      if (typeof data.detail === 'string') {
        return data.detail;
      }
      if (Array.isArray(data.detail)) {
        return data.detail.map((err: any) => err.msg).join(', ');
      }
    }
    if (data?.message) {
      return data.message;
    }
  } catch (_) {
    // ignore JSON parse errors
  }
  return response.statusText || 'Unknown API error';
};

async function apiCall<T>(
  endpoint: string,
  method: HttpMethod = 'GET',
  body?: unknown
): Promise<T> {
  const options: RequestInit = {
    method,
    headers: {
      'Content-Type': 'application/json'
    }
  };

  if (body !== undefined) {
    options.body = JSON.stringify(body);
  }

  const response = await fetch(`${API_BASE_URL}${endpoint}`, options);

  if (!response.ok) {
    const message = await parseErrorMessage(response);
    throw new Error(message);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  const contentType = response.headers.get('Content-Type');
  if (contentType && contentType.includes('application/json')) {
    return response.json();
  }

  return undefined as T;
}

async function fetchProductInfo(productId: number): Promise<Product> {
  const product = await apiCall<any>(`/recommendations/products/${productId}`, 'GET');
  return normalizeProduct(product);
}

async function normalizeCartResponse(response: ApiCartResponse): Promise<CartResponse> {
  const cartItems: CartItem[] = await Promise.all(
    (response.cart_items || []).map(async (item) => {
      const source = item.product ?? item.product_info;
      let product: Product;

      if (source) {
        product = normalizeProduct({ ...source, product_id: source.product_id ?? item.product_id });
      } else {
        product = await fetchProductInfo(item.product_id);
      }

      return {
        cart_item_id: item.cart_item_id,
        product_id: item.product_id,
        quantity: item.quantity,
        item_total: item.item_total ?? product.price * item.quantity,
        added_at: item.added_at,
        product
      };
    })
  );

  const totalQuantity = response.total_quantity ?? cartItems.reduce((sum, item) => sum + item.quantity, 0);
  const totalPrice = response.total_price ?? cartItems.reduce((sum, item) => sum + item.item_total, 0);

  return {
    user_id: response.user_id,
    cart_items: cartItems,
    total_items: response.total_items ?? cartItems.length,
    total_quantity: totalQuantity,
    total_price: totalPrice
  };
}

export const apiService = {
  async registerUser(userData: UserRegistration): Promise<UserRegistrationResponse> {
    const response = await apiCall<UserRegistrationResponse>('/user/register', 'POST', userData);
    localStorage.setItem('currentUser', JSON.stringify(response));
    return response;
  },

  async endSession(userId: number, sessionId: string): Promise<void> {
    await apiCall(`/user/${userId}/sessions/${sessionId}/end`, 'POST');
  },

  async getUserInfo(userId: number): Promise<User> {
    try {
      return await apiCall<User>(`/user/${userId}`, 'GET');
    } catch (error) {
      const storedUser = localStorage.getItem('currentUser');
      if (storedUser) {
        const parsed = JSON.parse(storedUser) as UserRegistrationResponse;
        return {
          user_id: parsed.user_id,
          name: parsed.name,
          age: parsed.age,
          income_level: parsed.income_level,
          price_sensitivity: parsed.profile_completed_with_medians?.price_sensitivity ?? 0,
          quality_sensitivity: parsed.profile_completed_with_medians?.quality_sensitivity ?? 0,
          exploration_tendency: parsed.profile_completed_with_medians?.exploration_tendency ?? 0
        };
      }
      throw error;
    }
  },

  async getRecommendations(userId: number, limit = 20): Promise<RecommendationsResponse> {
    const response = await apiCall<RecommendationsResponse>(
      `/recommendations/${userId}?limit=${Math.min(limit, 50)}`,
      'GET'
    );

    return {
      ...response,
      products: response.products.map(normalizeProduct)
    };
  },

  async getProductInfo(productId: number): Promise<Product> {
    return fetchProductInfo(productId);
  },

  async processAction(userId: number, action: UserAction): Promise<any> {
    return apiCall(`/${userId}/actions`, 'POST', {
      ...action,
      user_id: action.user_id ?? userId
    });
  },

  async getCart(userId: number): Promise<CartResponse> {
    const response = await apiCall<ApiCartResponse>(`/${userId}`, 'GET');
    return normalizeCartResponse(response);
  },

  async addToCart(userId: number, productId: number, quantity = 1): Promise<any> {
    return apiCall(`/${userId}/items`, 'POST', { product_id: productId, quantity });
  },

  async updateCartItem(userId: number, cartItemId: number, quantity: number): Promise<any> {
    return apiCall(`/${userId}/items/${cartItemId}?quantity=${quantity}`, 'PUT');
  },

  async removeFromCart(userId: number, cartItemId: number): Promise<any> {
    return apiCall(`/${userId}/items/${cartItemId}`, 'DELETE');
  },

  async clearCart(userId: number): Promise<any> {
    return apiCall(`/${userId}/clear`, 'DELETE');
  },

  async placeOrder(userId: number): Promise<any> {
    return apiCall(`/${userId}/order`, 'POST');
  }
};
