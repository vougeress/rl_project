// Product types
export interface Product {
  product_id: number;
  product_name: string;
  category_name: string;
  price: number;
  popularity: number;
  quality: number;
  name_format?: string | null;
  category_id?: number;
  style_vector?: Record<string, number> | number[];
  image_url?: string;
}

// User types
export interface User {
  user_id: number;
  name: string;
  age: number;
  gender?: string;
  income_level?: string;
  price_sensitivity?: number;
  quality_sensitivity?: number;
  exploration_tendency?: number;
}

export interface UserRegistration {
  name: string;
  age: number;
}

export interface UserRegistrationResponse {
  user_id: number;
  name: string;
  age: number;
  income_level: string;
  profile_completed_with_medians: Record<string, number>;
  message: string;
  session_id: string;
  session_started_at: string;
  session_number: number;
}

// Cart types
export interface CartItem {
  cart_item_id: number;
  product_id: number;
  quantity: number;
  item_total: number;
  added_at?: string;
  product: Product;
  product_info?: Product;
}

export interface CartResponse {
  user_id: number;
  cart_items: CartItem[];
  total_items: number;
  total_quantity: number;
  total_price: number;
}

// Action types
export type ActionType = 'view' | 'like' | 'dislike' | 'add_to_cart' | 'purchase' | 'share' | 'close_immediately' | 'report' | 'report_spam' | 'remove_from_cart';

export interface UserAction {
  user_id: number;
  product_id?: number;
  action_type: ActionType;
  session_id?: string;
  experiment_id?: string;
}

export interface UserActionResponse {
  success: boolean;
  action_id: number;
  session_id: string;
  reward: number;
  message: string;
}

// Recommendations
export interface RecommendationsResponse {
  products: Product[];
  user_id: number;
  total_count: number;
}

// Categories
export const CATEGORIES = [
  'Electronics',
  'Clothing',
  'Books',
  'Home & Garden',
  'Sports',
  'Beauty',
  'Toys',
  'Automotive',
  'Health',
  'Food'
] as const;

export type CategoryName = typeof CATEGORIES[number];
