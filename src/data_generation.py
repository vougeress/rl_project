"""
Data generation module for creating synthetic product catalog and user profiles.

IMPROVEMENTS FOR BETTER LEARNING:
1. Rewards now scale with base_engagement - better signal for learning
2. Reduced time drift (0.002 vs 0.01) - more stable preferences for learning
3. Increased action probabilities - more positive feedback for good recommendations
4. Better base_engagement range (0.15-0.95 vs 0.1-0.9) - more differentiation

User Action Types for Simulation:

POSITIVE ACTIONS (ENGAGEMENT-DEPENDENT REWARDS):
1. VIEW (Просмотр)
   - Probability: 100% (always occurs)
   - Reward: 0.1 + 0.2 × base_engagement (0.13-0.29) - scales with quality

2. LIKE (Добавить в избранное)
   - Probability: base_engagement × 0.5 (maximum 60%)
   - Reward: 0.8 + 0.4 × base_engagement (0.86-1.16) - scales with quality

3. ADD_TO_CART (Добавить в корзину)
   - Probability: (base_engagement - 0.15) × 0.4 (maximum 30%)
   - Reward: 2.5 + 1.0 × base_engagement (2.65-3.45) - scales with quality

4. PURCHASE (Покупка)
   - Probability: (base_engagement - 0.25) × 0.25 × price_match (maximum 20%)
   - Reward: 7.0 + 2.0 × base_engagement (7.3-8.9) - scales with quality

5. SHARE (Поделиться)
   - Probability: quality × popularity × base_engagement × 0.12 (maximum 10%)
   - Reward: 3.5 + 1.0 × base_engagement (3.65-4.45) - scales with quality

NEGATIVE ACTIONS (ENGAGEMENT-DEPENDENT PENALTIES):
1. DISLIKE (Не понравилось)
   - Probability: (1 - base_engagement) × 0.15 (maximum 25%)
   - Reward: -0.2 - 0.2 × (1 - base_engagement) (-0.2 to -0.37)

2. REPORT_SPAM (Жалоба)
   - Probability: (1 - base_engagement) × (1 - quality) × 0.001 (maximum 0.2%)
   - Reward: -1.0 (moderate penalty)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import random
from enum import Enum
from dataclasses import dataclass


@dataclass
class ActionSpec:
    """Specification for a user action type."""
    name: str
    max_probability: float
    reward: float
    description: str


class UserActionTypes:
    """User action types for simulation with their specifications."""

    # Positive Actions
    VIEW = ActionSpec(
        name="view",
        max_probability=1.0,  # Always occurs
        reward=0.3,  # Increased from 0.1
        description="User viewed the product"
    )

    LIKE = ActionSpec(
        name="like",
        max_probability=0.50,  # Increased from 15% to 50%
        reward=1.0,  # Increased from 0.5
        description="User added to favorites"
    )

    ADD_TO_CART = ActionSpec(
        name="add_to_cart",
        max_probability=0.25,  # Increased from 8% to 25%
        reward=3.0,  # Increased from 2.0
        description="User added to cart"
    )

    PURCHASE = ActionSpec(
        name="purchase",
        max_probability=0.15,  # Increased from 3% to 15%
        reward=8.0,  # Decreased from 10.0 for balance
        description="User purchased the product"
    )

    SHARE = ActionSpec(
        name="share",
        max_probability=0.08,  # Increased from 1% to 8%
        reward=4.0,  # Increased from 3.0
        description="User shared the product"
    )

    # Negative Actions
    DISLIKE = ActionSpec(
        name="dislike",
        max_probability=0.25,  # Decreased from 40% to 25%
        reward=-0.5,  # Increased penalty from -0.2
        description="User disliked the product"
    )

    REPORT_SPAM = ActionSpec(
        name="report_spam",
        max_probability=0.005,  # Increased from 0.1% to 0.5%
        reward=-2.0,  # Increased penalty from -1.0
        description="User reported product as spam"
    )

    @classmethod
    def get_all_actions(cls):
        """Get all action specifications."""
        return [
            cls.VIEW, cls.LIKE, cls.ADD_TO_CART, cls.PURCHASE,
            cls.SHARE, cls.DISLIKE, cls.REPORT_SPAM
        ]

    @classmethod
    def get_positive_actions(cls):
        """Get positive action specifications."""
        return [cls.VIEW, cls.LIKE, cls.ADD_TO_CART, cls.PURCHASE, cls.SHARE]

    @classmethod
    def get_negative_actions(cls):
        """Get negative action specifications."""
        return [cls.DISLIKE, cls.REPORT_SPAM]


class ProductCatalog:
    """Generate and manage synthetic product catalog."""

    def __init__(self, n_products: int = 1000, n_categories: int = 10, style_dim: int = 5):
        self.n_products = n_products
        self.n_categories = n_categories
        self.style_dim = style_dim
        self.products_df = None
        self.category_names = [
            'Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports',
            'Beauty', 'Toys', 'Automotive', 'Health', 'Food'
        ][:n_categories]

    def generate_products(self) -> pd.DataFrame:
        """Generate synthetic product catalog."""
        np.random.seed(42)

        products = []
        used_names = set()
        for i in range(self.n_products):
            while True:
                category_id = np.random.randint(0, self.n_categories)
                category_name = self.category_names[category_id]

                product_name, name_format = self._generate_product_name(category_name)

                if product_name not in used_names:
                    used_names.add(product_name)
                    break

            # Price varies by category
            base_prices = {
                'Electronics': 200, 'Clothing': 50, 'Books': 15,
                'Home & Garden': 80, 'Sports': 60, 'Beauty': 30,
                'Toys': 25, 'Automotive': 150, 'Health': 40, 'Food': 10
            }
            base_price = base_prices.get(category_name, 50)

            # Quality score (affects engagement)
            if np.random.rand() < 0.1:  # 10% premium
                quality = np.random.beta(5, 2)
            else:
                quality = np.random.beta(2, 5)

            # Popularity score (0-1)
            popularity = np.clip(np.random.beta(2, 5) + 0.3 * quality, 0, 1)

            price = base_price * (0.5 + quality) * (0.8 + popularity)
            price = max(5, np.random.lognormal(np.log(price), 0.3))

            # Style vector (normalized)
            style_vector = np.random.randn(self.style_dim)
            style_vector = style_vector / np.linalg.norm(style_vector)

            product = {
                'product_id': i,
                'product_name': product_name,
                'name_format': name_format,
                'category_id': category_id,
                'category_name': category_name,
                'price': price,
                'popularity': popularity,
                'quality': quality,
                **{f'style_{j}': style_vector[j] for j in range(self.style_dim)}
            }
            products.append(product)

        self.products_df = pd.DataFrame(products)
        return self.products_df

    def _generate_product_name(self, category: str) -> tuple[str, str]:
        category_data = {
            'Electronics': {
                'brands': ['Samsung', 'Apple', 'Sony', 'LG', 'Xiaomi', 'Huawei', 'Dell', 'Canon'],
                'types': ['Smartphone', 'Laptop', 'Tablet', 'TV', 'Camera', 'Headphones', 'Speaker', 'Smartwatch'],
                'adjectives': ['Pro', 'Max', 'Ultra', 'Lite', 'Plus', 'Elite', 'Premium', 'Wireless'],
                'features': ['4K', 'Bluetooth', 'Wi-Fi', 'OLED', 'HD', 'Smart', 'Portable', 'Gaming'],
                'models': ['X', 'Pro', 'Max', 'Ultra', 'Lite', 'Plus', 'Elite'],
                'formats': [
                    "{brand} {feature} {type} {model}-{number}",
                    "{brand} {type} {model} {feature}",
                    "{brand} {adjective} {type}",
                    "{feature} {type} {model} by {brand}"
                ]
            },
            'Clothing': {
                'brands': ['Nike', 'Adidas', 'Zara', 'H&M', 'Uniqlo', 'Levis', 'Puma', 'Gucci'],
                'types': ['T-Shirt', 'Jeans', 'Dress', 'Jacket', 'Sweater', 'Shorts', 'Skirt', 'Coat'],
                'adjectives': ['Classic', 'Premium', 'Comfort', 'Sport', 'Designer', 'Casual', 'Slim', 'Oversized'],
                'features': ['Cotton', 'Denim', 'Wool', 'Linen', 'Stretch', 'Waterproof', 'Breathable'],
                'formats': [
                    "{brand} {adjective} {type}",
                    "{feature} {type} by {brand}",
                    "{brand} {type}",
                    "{adjective} {feature} {type}"
                ]
            },
            'Books': {
                'brands': ['Penguin', 'HarperCollins', 'Random House', 'Simon & Schuster', 'Macmillan'],
                'types': ['Novel', 'Guide', 'Textbook', 'Biography', 'Cookbook', 'Journal', 'Dictionary'],
                'adjectives': ['Complete', 'Essential', 'Advanced', 'Beginner', 'Illustrated', 'Digital'],
                'topics': ['Programming', 'History', 'Science', 'Art', 'Business', 'Psychology', 'Cooking'],
                'formats': [
                    "{adjective} {topic} {type}",
                    "The {topic} {type}",
                    "{topic}: {adjective} {type}",
                    "{brand} {topic} {type}"
                ]
            },
            'Home & Garden': {
                'brands': ['IKEA', 'HomeDepot', 'Williams-Sonoma', 'Crate&Barrel', 'Wayfair'],
                'types': ['Sofa', 'Table', 'Chair', 'Lamp', 'Tool Set', 'Plant', 'Decoration', 'Cookware'],
                'adjectives': ['Modern', 'Vintage', 'Rustic', 'Minimalist', 'Luxury', 'Eco-Friendly'],
                'materials': ['Wood', 'Metal', 'Glass', 'Ceramic', 'Bamboo', 'Marble'],
                'formats': [
                    "{adjective} {material} {type}",
                    "{brand} {material} {type}",
                    "{adjective} {type} by {brand}",
                    "{material} {type}"
                ]
            },
            'Sports': {
                'brands': ['Nike', 'Adidas', 'Under Armour', 'Reebok', 'Puma', 'Wilson', 'Spalding'],
                'types': ['Basketball', 'Sneakers', 'Yoga Mat', 'Dumbbells', 'Tennis Racket', 'Football', 'Jacket'],
                'adjectives': ['Pro', 'Elite', 'Training', 'Performance', 'Competition', 'Premium'],
                'formats': [
                    "{brand} {type}",
                    "{adjective} {type}",
                    "{brand} {adjective} {type}",
                    "{type} by {brand}"
                ]
            },
            'Beauty': {
                'brands': ['L-Oreal', 'Maybelline', 'Nivea', 'Olay', 'Neutrogena', 'Dove'],
                'types': ['Shampoo', 'Conditioner', 'Moisturizer', 'Foundation', 'Lipstick', 'Perfume', 'Serum'],
                'adjectives': ['Hydrating', 'Anti-Aging', 'Nourishing', 'Professional', 'Luxury', 'Organic'],
                'benefits': ['Hydration', 'Repair', 'Protection', 'Revitalizing', 'Whitening'],
                'formats': [
                    "{brand} {adjective} {type}",
                    "{benefits} {type} by {brand}",
                    "{adjective} {benefits} {type}",
                    "{brand} {type}"
                ]
            },
            'Toys': {
                'brands': ['Lego', 'Hasbro', 'Mattel', 'Fisher-Price', 'Playmobil'],
                'types': ['Action Figure', 'Puzzle', 'Board Game', 'Doll', 'RC Car', 'Plush Toy'],
                'adjectives': ['Fun', 'Interactive', 'Educational', 'Deluxe', 'Mini', 'Collectible'],
                'formats': [
                    "{brand} {adjective} {type}",
                    "{type} by {brand}",
                    "{adjective} {type} {brand}"
                ]
            },
            'Automotive': {
                'brands': ['Toyota', 'Ford', 'BMW', 'Honda', 'Chevrolet', 'Tesla', 'Audi'],
                'types': ['Tire', 'Engine Oil', 'Car Battery', 'Brake Pads', 'Air Filter', 'GPS', 'Car Seat'],
                'adjectives': ['Premium', 'Durable', 'Performance', 'Eco', 'Sport', 'Advanced'],
                'formats': [
                    "{brand} {type}",
                    "{adjective} {type} by {brand}",
                    "{type} {brand}"
                ]
            },
            'Health': {
                'brands': ['NatureMade', 'Centrum', 'GNC', 'Solgar'],
                'types': ['Vitamin', 'Supplement', 'Protein Powder', 'Mineral', 'Herbal Tea'],
                'adjectives': ['Natural', 'Organic', 'High Potency', 'Daily', 'Advanced'],
                'formats': [
                    "{brand} {adjective} {type}",
                    "{type} by {brand}",
                    "{adjective} {type}"
                ]
            },
            'Food': {
                'brands': ['Nestle', 'Kraft', 'Heinz', 'Danone', 'PepsiCo', 'Mars'],
                'types': ['Chocolate', 'Cereal', 'Snack', 'Beverage', 'Sauce', 'Cookies'],
                'adjectives': ['Delicious', 'Organic', 'Premium', 'Healthy', 'Tasty'],
                'formats': [
                    "{brand} {adjective} {type}",
                    "{adjective} {type} by {brand}",
                    "{type} {brand}"
                ]
            }
        }

        default_data = {
            'brands': ['Alpha', 'Beta', 'Gamma', 'Delta', 'Omega'],
            'types': ['Product', 'Item', 'Goods'],
            'adjectives': ['Premium', 'Quality', 'Standard', 'Advanced'],
            'formats': ["{brand} {adjective} {type}"]
        }

        data = category_data.get(category, default_data)

        brand = np.random.choice(data['brands'])
        product_type = np.random.choice(data['types'])
        adjective = np.random.choice(data['adjectives'])

        format_template = np.random.choice(data['formats'])

        name = format_template.format(
            brand=brand,
            type=product_type,
            adjective=adjective,
            feature=np.random.choice(data.get('features', [''])),
            model=np.random.choice(data.get('models', [''])),
            number=np.random.randint(10, 25),
            collection=np.random.choice(data.get('collections', [''])),
            topic=np.random.choice(data.get('topics', [''])),
            material=np.random.choice(data.get('materials', [''])),
            sports=np.random.choice(data.get('sports', [''])),
            benefits=np.random.choice(data.get('benefits', ['']))
        )

        return name, format_template

    def get_product_features(self, product_id: int) -> np.ndarray:
        """Get feature vector for a specific product."""
        if self.products_df is None:
            raise ValueError("Products not generated yet. Call generate_products() first.")

        product = self.products_df.iloc[product_id]
        features = np.array([
            product['category_id'] / self.n_categories,  # Normalized category
            np.log(product['price']) / 10,  # Log-normalized price
            product['popularity'],
            product['quality'],
            *[product[f'style_{j}'] for j in range(self.style_dim)]
        ])
        return features

    def get_product_info(self, product_id: int) -> Dict:
        """Get detailed product information."""
        if self.products_df is None:
            raise ValueError("Products not generated yet. Call generate_products() first.")

        if product_id < 0 or product_id >= len(self.products_df):
            raise ValueError(f"Invalid product_id {product_id}")

        product = self.products_df.iloc[product_id]
        return {
            'product_id': product_id,
            'product_name': product['product_name'],
            'name_format': product.get('name_format'),
            'category_name': product['category_name'],
            'category': product['category_name'],
            'price': float(product['price']),
            'popularity': float(product['popularity']),
            'quality': float(product['quality'])
        }


class UserSimulator:
    """Simulate users with evolving preferences."""

    def __init__(self, n_users: int = 100, style_dim: int = 5, n_categories: int = 10):
        self.n_users = n_users
        self.style_dim = style_dim
        self.n_categories = n_categories
        self.users_df = None

    def generate_users(self) -> pd.DataFrame:
        """Generate synthetic user profiles."""
        np.random.seed(123)

        users = []
        for i in range(self.n_users):
            # Age affects preferences
            age = np.random.randint(18, 70)
            age_normalized = (age - 18) / (70 - 18)

            # Budget affects price sensitivity
            if age < 25:
                income_level = np.random.choice(['low', 'medium', 'high'], p=[0.65, 0.3, 0.05])
            elif age < 55:
                income_level = np.random.choice(['low', 'medium', 'high'], p=[0.2, 0.6, 0.2])
            else:
                income_level = np.random.choice(['low', 'medium', 'high'], p=[0.4, 0.2, 0.4])

            budget_multiplier = {'low': 0.5, 'medium': 1.0, 'high': 2.0}[income_level]

            # Category preferences (some users prefer certain categories)
            category_weights = np.ones(self.n_categories)

            if age < 30:
                category_weights[[0, 1, 4, 5]] *= 1.5

            elif age < 50:
                category_weights[[3, 4, 6, 8]] *= 1.5

            else:
                category_weights[[2, 3, 8]] *= 1.5

            category_preferences = np.random.dirichlet(category_weights)

            # Style preferences (normalized vector)
            style_preferences = np.random.randn(self.style_dim)
            style_preferences = style_preferences / np.linalg.norm(style_preferences)

            user_type = np.random.choice(['explorer', 'thrifty', 'premium', 'trend_follower'])
            if user_type == 'explorer':
                base_exploration = np.random.beta(5, 2)
                base_price_sens = np.random.beta(2, 5)
                base_quality_sens = np.random.beta(3, 2)
            elif user_type == 'thrifty':
                base_exploration = np.random.beta(2, 5)
                base_price_sens = np.random.beta(5, 2)
                base_quality_sens = np.random.beta(3, 2)
            elif user_type == 'premium':
                base_exploration = np.random.beta(3, 2)
                base_price_sens = np.random.beta(2, 5)
                base_quality_sens = np.random.beta(5, 2)
            else:
                base_exploration = np.random.beta(3, 2)
                base_price_sens = np.random.beta(3, 2)
                base_quality_sens = np.random.beta(3, 2)

            price_sensitivity = np.clip(base_price_sens * (1.5 - budget_multiplier), 0, 1)
            exploration_tendency = np.clip(base_exploration * (1.2 - age_normalized), 0, 1)
            quality_sensitivity = np.clip(base_quality_sens * (0.5 + 0.5 * budget_multiplier), 0, 1)

            user = {
                'user_id': i,
                'age': age,
                'age_normalized': age_normalized,
                'income_level': income_level,
                'budget_multiplier': budget_multiplier,
                'price_sensitivity': price_sensitivity,
                'quality_sensitivity': quality_sensitivity,
                'exploration_tendency': exploration_tendency,
                **{f'category_pref_{j}': category_preferences[j] for j in range(self.n_categories)},
                **{f'style_pref_{j}': style_preferences[j] for j in range(self.style_dim)}
            }
            users.append(user)

        self.users_df = pd.DataFrame(users)
        return self.users_df

    def get_user_state(self, user_id: int, session_time: int = 0) -> np.ndarray:
        """Get current user state vector (includes time-based drift)."""
        if self.users_df is None:
            raise ValueError("Users not generated yet. Call generate_users() first.")

        user = self.users_df.iloc[user_id]

        # Base preferences
        style_prefs = np.array([user[f'style_pref_{j}'] for j in range(self.style_dim)])
        category_prefs = np.array([user[f'category_pref_{j}'] for j in range(self.n_categories)])

        # Add time-based drift (preferences evolve slowly)
        # Используем нелинейный дрейф - быстрее в начале, медленнее потом
        # Это позволяет агентам быстрее адаптироваться к изменениям
        if session_time > 0:
            # Логарифмический дрейф - замедляется со временем
            drift_factor = 0.001 * np.log1p(session_time)  # Логарифмический рост вместо линейного
            style_drift = np.random.randn(self.style_dim) * drift_factor
            style_prefs = style_prefs + style_drift
            style_prefs = style_prefs / np.linalg.norm(style_prefs)  # Renormalize

        # Combine all features into state vector
        state = np.concatenate([
            [user['age_normalized']],
            [user['budget_multiplier'] / 2.0],  # Normalize to [0, 1]
            [user['price_sensitivity']],
            [user['quality_sensitivity']],
            [user['exploration_tendency']],
            category_prefs,
            style_prefs,
            [session_time / 100.0]  # Normalized session time
        ])

        return state

    def calculate_user_actions(self, user_id: int, product_features: np.ndarray,
                              session_time: int = 0) -> dict:
        """Calculate probabilities and rewards for different user actions."""
        user = self.users_df.iloc[user_id]

        # Extract product features
        category_id = int(product_features[0] * self.n_categories)
        log_price = product_features[1] * 10
        price = np.exp(log_price)
        popularity = product_features[2]
        quality = product_features[3]
        product_style = product_features[4:4+self.style_dim]

        # User preferences
        user_style = np.array([user[f'style_pref_{j}'] for j in range(self.style_dim)])
        category_pref = user[f'category_pref_{category_id}']

        # Calculate base engagement factors

        # 1. Style match (cosine similarity)
        style_match = np.dot(user_style, product_style)

        # 2. Category preference
        category_match = category_pref

        # 3. Price appropriateness
        ideal_price = 50 * user['budget_multiplier']
        price_match = np.exp(-((price - ideal_price) ** 2) / (2 * (ideal_price * 0.5) ** 2))
        price_penalty = user['price_sensitivity'] * (1 - price_match)

        # 4. Quality appreciation
        quality_bonus = user['quality_sensitivity'] * quality

        # 5. Popularity effect
        popularity_bonus = (1 - user['exploration_tendency']) * popularity

        # Base engagement score - улучшенная формула для лучшей дифференциации
        base_engagement = (
            0.35 * max(0, style_match) +
            0.30 * category_match +
            0.20 * max(0, 1 - price_penalty) +
            0.10 * quality_bonus +
            0.05 * popularity_bonus
        )
        base_engagement = np.clip(base_engagement, 0.15, 0.95)  # Минимум 15%, максимум 95% (улучшен диапазон)

        # Calculate action probabilities and rewards
        actions = {}

        # 1. VIEW (просмотр) - базовое действие, всегда происходит
        view_reward = 0.1 + 0.2 * base_engagement
        actions['view'] = {
            'probability': 1.0,  # Всегда происходит
            'reward': view_reward,  # Награда зависит от качества рекомендации
            'description': 'Пользователь просмотрел товар'
        }

        # 2. LIKE (добавить в избранное)
        like_prob = base_engagement * 0.5 + np.random.normal(0, 0.02)
        actions['like'] = {
            'probability': np.clip(like_prob, 0, 0.60),
            'reward': 0.8 + 0.4 * base_engagement,
            'description': 'Пользователь добавил в избранное'
        }

        # 3. ADD_TO_CART (добавить в корзину) - увеличена вероятность
        cart_prob = (base_engagement - 0.15) * 0.4 + np.random.normal(0, 0.01)  # Снижен порог, увеличена вероятность
        actions['add_to_cart'] = {
            'probability': np.clip(cart_prob, 0, 0.30),  # Максимум 30%
            'reward': 2.5 + 1.0 * base_engagement,  # Награда от 2.6 до 3.4
            'description': 'Пользователь добавил в корзину'
        }

        # 4. PURCHASE (покупка) - увеличена вероятность
        purchase_prob = (base_engagement - 0.25) * 0.25 * price_match + np.random.normal(0, 0.005)  # Снижен порог
        actions['purchase'] = {
            'probability': np.clip(purchase_prob, 0, 0.20),  # Максимум 20%
            'reward': 7.0 + 2.0 * base_engagement,  # Награда от 7.2 до 8.8
            'description': 'Пользователь купил товар'
        }

        # 5. SHARE (поделиться) - увеличена вероятность и зависимость от engagement
        share_prob = quality * popularity * base_engagement * 0.12 + np.random.normal(0, 0.002)  # Увеличена вероятность
        actions['share'] = {
            'probability': np.clip(share_prob, 0, 0.10),  # Максимум 10%
            'reward': 3.5 + 1.0 * base_engagement,  # Награда от 3.65 до 4.45
            'description': 'Пользователь поделился товаром'
        }

        # НЕГАТИВНЫЕ ДЕЙСТВИЯ - более реалистичные вероятности

        # 6. DISLIKE (не понравилось) - умеренный штраф, зависит от base_engagement
        dislike_prob = (1 - base_engagement) * 0.15 + np.random.normal(0, 0.02)  # Снижена вероятность
        actions['dislike'] = {
            'probability': np.clip(dislike_prob, 0, 0.25),  # Максимум 25%
            'reward': -0.2 - 0.2 * (1 - base_engagement),  # Штраф от -0.2 до -0.38
            'description': 'Пользователю не понравился товар'
        }

        # 7. REPORT_SPAM (пожаловаться) - очень редко, умеренный штраф
        spam_prob = (1 - base_engagement) * (1 - quality) * 0.001 + np.random.normal(0, 0.0005)
        actions['report_spam'] = {
            'probability': np.clip(spam_prob, 0, 0.002),  # Максимум 0.2%
            'reward': -1.0,  # Умеренный штраф
            'description': 'Пользователь пожаловался на товар'
        }

        return actions

    def calculate_engagement(self, user_id: int, product_features: np.ndarray,
                           session_time: int = 0) -> float:
        """Calculate user engagement score for a product (backward compatibility)."""
        actions = self.calculate_user_actions(user_id, product_features, session_time)

        # Возвращаем взвешенную сумму всех положительных действий
        total_engagement = 0
        for action_name, action_data in actions.items():
            if action_data['reward'] > 0:
                total_engagement += action_data['probability'] * action_data['reward']

        return np.clip(total_engagement, 0, 1)

    def simulate_user_interaction(self, user_id: int, product_features: np.ndarray,
                                 session_time: int = 0) -> dict:
        """Simulate actual user interaction with a product."""
        actions = self.calculate_user_actions(user_id, product_features, session_time)

        # Симулируем какие действия произойдут
        occurred_actions = []
        total_reward = 0

        for action_name, action_data in actions.items():
            if np.random.random() < action_data['probability']:
                occurred_actions.append({
                    'action': action_name,
                    'reward': action_data['reward'],
                    'description': action_data['description']
                })
                total_reward += action_data['reward']

        return {
            'occurred_actions': occurred_actions,
            'total_reward': total_reward,
            'all_action_probs': {name: data['probability'] for name, data in actions.items()}
        }

    def get_user_info(self, user_id: int) -> Dict:
        """Get user information."""
        if self.users_df is None:
            raise ValueError("Users not generated yet. Call generate_users() first.")

        if user_id < 0 or user_id >= len(self.users_df):
            raise ValueError(f"Invalid user_id {user_id}")

        user = self.users_df.iloc[user_id]
        return {
            'user_id': user_id,
            'age': int(user['age']),
            'income_level': user['income_level'],
            'price_sensitivity': float(user['price_sensitivity']),
            'quality_sensitivity': float(user['quality_sensitivity']),
            'exploration_tendency': float(user['exploration_tendency'])
        }


def generate_synthetic_data(n_products: int = 1000, n_users: int = 100,
                          n_categories: int = 10, style_dim: int = 5) -> Tuple[ProductCatalog, UserSimulator]:
    """Generate complete synthetic dataset."""

    # Generate products
    catalog = ProductCatalog(n_products, n_categories, style_dim)
    products_df = catalog.generate_products()
    
    # Generate users
    simulator = UserSimulator(n_users, style_dim, n_categories)
    users_df = simulator.generate_users()

    print(f"Generated {len(products_df)} products and {len(users_df)} users")
    print(f"Product categories: {catalog.category_names}")
    print(f"Style dimensions: {style_dim}")

    return catalog, simulator


if __name__ == "__main__":
    # Test data generation
    catalog, simulator = generate_synthetic_data()

    # Test product features
    product_features = catalog.get_product_features(0)
    print(f"Product 0 features: {product_features}")

    # Test user state
    user_state = simulator.get_user_state(0)
    print(f"User 0 state: {user_state}")

    # Test engagement calculation
    engagement = simulator.calculate_engagement(0, product_features)
    print(f"User 0 engagement with Product 0: {engagement:.3f}")
