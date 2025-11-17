from sqlalchemy import Column, Index, Integer, String, Float, DateTime, Text, CheckConstraint, ForeignKey, JSON
from sqlalchemy.sql import func
from api.core.database import Base

class User(Base):
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    income_level = Column(String(20), nullable=False)
    price_sensitivity = Column(Float, default=0.5)
    quality_sensitivity = Column(Float, default=0.5)
    exploration_tendency = Column(Float, default=0.5)
    budget_multiplier = Column(Float, default=1.0)
    category_preferences = Column(JSON)
    style_preferences = Column(JSON)
    state_vector = Column(JSON)
    registered_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        CheckConstraint('age >= 16 AND age <= 100', name='age_check'),
        CheckConstraint("income_level IN ('low', 'medium', 'high')", name='income_level_check'),
        CheckConstraint('price_sensitivity >= 0 AND price_sensitivity <= 1', name='price_sensitivity_check'),
        CheckConstraint('quality_sensitivity >= 0 AND quality_sensitivity <= 1', name='quality_sensitivity_check'),
        CheckConstraint('exploration_tendency >= 0 AND exploration_tendency <= 1', name='exploration_tendency_check'),
        CheckConstraint('budget_multiplier >= 0.5 AND budget_multiplier <= 2.0', name='budget_multiplier_check'),
        Index('idx_users_income', 'income_level'),
        Index('idx_users_age', 'age'),
    )

class Product(Base):
    __tablename__ = "products"
    
    product_id = Column(Integer, primary_key=True, index=True)
    category_id = Column(Integer, nullable=False)
    category_name = Column(String(50), nullable=False)
    price = Column(Float, nullable=False)
    popularity = Column(Float, default=0.5)
    quality = Column(Float, default=0.5)
    style_vector = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        CheckConstraint('category_id >= 0 AND category_id <= 9', name='category_id_check'),
        CheckConstraint("category_name IN ('Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports', 'Beauty', 'Toys', 'Automotive', 'Health', 'Food')", name='category_name_check'),
        CheckConstraint('price > 0', name='price_check'),
        CheckConstraint('popularity >= 0 AND popularity <= 1', name='popularity_check'),
        CheckConstraint('quality >= 0 AND quality <= 1', name='quality_check'),
        Index('idx_products_category', 'category_name'),
        Index('idx_products_price', 'price'),
        Index('idx_products_popularity', 'popularity'),
    )

class Experiment(Base):
    __tablename__ = "experiments"
    
    experiment_id = Column(String(20), primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    status = Column(String(20), default='pending')
    progress = Column(Float, default=0.0)
    configuration = Column(JSON)
    agent_type = Column(String(20), nullable=False)
    agent_params = Column(JSON)
    results = Column(JSON)
    start_time = Column(DateTime(timezone=True))
    end_time = Column(DateTime(timezone=True))
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'running', 'completed', 'failed', 'cancelled')", name='status_check'),
        CheckConstraint('progress >= 0 AND progress <= 1', name='progress_check'),
        CheckConstraint("agent_type IN ('dqn', 'linucb', 'epsilon_greedy', 'random')", name='agent_type_check'),
        Index('idx_experiments_status', 'status'),
        Index('idx_experiments_agent', 'agent_type'),
    )

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    session_id = Column(String(50), primary_key=True)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True))
    session_length = Column(Integer, default=0)
    total_reward = Column(Float, default=0.0)
    actions_count = Column(Integer, default=0)
    experiment_id = Column(String(20), ForeignKey('experiments.experiment_id'))

    __table_args__ = (
        Index('idx_sessions_user', 'user_id'),
        Index('idx_sessions_time', 'start_time'),
    )

class UserAction(Base):
    __tablename__ = "user_actions"
    
    action_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    product_id = Column(Integer, ForeignKey('products.product_id', ondelete='CASCADE'), nullable=False)
    action_type = Column(String(30), nullable=False)
    reward = Column(Float, nullable=False)
    session_time = Column(Integer, default=0)
    style_match = Column(Float, nullable=True)
    category_match = Column(Float, nullable=True)
    price_match = Column(Float, nullable=True)
    quality_bonus = Column(Float, nullable=True)
    popularity_bonus = Column(Float, nullable=True)
    action_probability = Column(Float, nullable=True)
    action_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    session_id = Column(String(50))
    experiment_id = Column(String(20), ForeignKey('experiments.experiment_id'))
    
    __table_args__ = (
        CheckConstraint("action_type IN ('view', 'like', 'add_to_cart', 'purchase', 'share', 'dislike', 'close_immediately', 'report', 'remove_from_cart')", name='action_type_check'),
        Index('idx_user_actions_user_id', 'user_id'),
        Index('idx_user_actions_timestamp', 'action_timestamp'),
        Index('idx_user_actions_session', 'session_id'),
        Index('idx_user_actions_type', 'action_type'),
        Index('idx_user_actions_experiment', 'experiment_id'),
    )

class CartItem(Base):
    __tablename__ = "cart_items"
    
    cart_item_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    product_id = Column(Integer, ForeignKey('products.product_id'))
    quantity = Column(Integer, default=1)
    added_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        CheckConstraint('quantity > 0', name='quantity_check'),
        Index('idx_cart_items_user_id', 'user_id'),
    )

class Order(Base):
    __tablename__ = "orders"
    
    order_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    total_price = Column(Float, nullable=False)
    status = Column(String(20), default='confirmed')
    order_date = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    items = Column(JSON)
    
    __table_args__ = (
        CheckConstraint('total_price > 0', name='total_price_check'),
        CheckConstraint("status IN ('pending', 'confirmed', 'completed', 'cancelled')", name='order_status_check'),
        Index('idx_orders_user_id', 'user_id'),
        Index('idx_orders_date', 'order_date'),
    )