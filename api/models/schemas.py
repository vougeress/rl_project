"""
Pydantic models for API requests and responses.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Any
from datetime import datetime


class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ExperimentConfig(BaseModel):
    n_products: int = 500
    n_users: int = 50
    n_episodes: int = 200
    max_session_length: int = 20
    agents: List[str] = ["epsilon_greedy", "linucb", "dqn", "random"]
    agent_params: Dict[str, Dict] = {}


class ExperimentBase(BaseSchema):
    experiment_id: str
    name: str
    description: Optional[str] = None
    status: str
    progress: float
    agent_type: str
    configuration: Optional[Dict[str, Any]] = None
    agent_params: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None


class ExperimentCreate(BaseSchema):
    name: str
    description: Optional[str] = None
    configuration: Dict[str, Any]
    agent_type: str
    agent_params: Dict[str, Any]


class ExperimentStatus(BaseSchema):
    experiment_id: str
    name: str
    description: Optional[str] = None
    agent_type: Optional[str] = None
    status: str  # "running", "completed", "failed"
    progress: float  # 0.0 to 1.0
    current_agent: Optional[str] = None
    configuration: Optional[Dict[str, Any]] = None
    agent_params: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    created_at: Optional[datetime] = None
    error: Optional[str] = None


class UserBase(BaseSchema):
    user_id: int
    name: str
    age: int
    income_level: str
    price_sensitivity: float
    quality_sensitivity: float
    exploration_tendency: float
    budget_multiplier: float
    category_preferences: Optional[Dict[str, Any]] = None
    style_preferences: Optional[Dict[str, Any]] = None
    state_vector: Optional[Dict[str, Any]] = None
    registered_at: datetime


class UserRegistration(BaseSchema):
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=16, le=100)


class UserRegistrationResponse(BaseSchema):
    user_id: int
    name: str
    age: int
    income_level: str
    profile_completed_with_medians: Dict[str, float]
    message: str
    session_id: str
    session_started_at: datetime
    session_number: int


class UserInfo(BaseSchema):
    user_id: int
    name: str
    age: int
    income_level: str
    price_sensitivity: float
    quality_sensitivity: float
    exploration_tendency: float


class ProductBase(BaseSchema):
    product_id: int
    product_name: str
    name_format: Optional[str] = None
    category_id: int
    category_name: str
    price: float
    popularity: float
    quality: float
    style_vector: Optional[Dict[str, Any]] = None
    created_at: datetime


class ProductInfo(BaseSchema):
    product_id: int
    product_name: str
    name_format: Optional[str] = None
    category_name: str
    price: float
    popularity: float
    quality: float


class RecommendationRequest(BaseSchema):
    user_id: int
    session_time: int = 0
    agent_type: str = "dqn"
    experiment_id: Optional[str] = None


class RecommendationsResponse(BaseSchema):
    products: List[ProductInfo]
    user_id: int
    total_count: int


class UserActionResult(BaseSchema):
    action: str
    reward: float
    description: str


class UserActionBase(BaseSchema):
    user_id: int
    product_id: Optional[int] = None
    action_type: str
    reward: float
    session_time: int = 0
    style_match: Optional[float] = None
    category_match: Optional[float] = None
    price_match: Optional[float] = None
    quality_bonus: Optional[float] = None
    popularity_bonus: Optional[float] = None
    action_probability: Optional[float] = None
    session_id: Optional[str] = None
    experiment_id: Optional[str] = None


class UserActionCreate(BaseSchema):
    user_id: int
    product_id: Optional[int] = None
    action_type: str = Field(
        ...,
        pattern="^(view|like|add_to_cart|purchase|share|dislike|close_immediately|report|report_spam|remove_from_cart)$"
    )
    session_id: Optional[str] = None
    experiment_id: Optional[str] = None


class ProductActionResponse(BaseSchema):
    user_id: int
    product_id: int
    product_info: ProductInfo
    action: str
    message: str
    reward: float
    timestamp: datetime


class InteractionResult(BaseSchema):
    occurred_actions: List[UserActionResult]
    total_reward: float
    action_probabilities: Dict[str, float]


class CartItemBase(BaseSchema):
    cart_item_id: int
    user_id: int
    product_id: int
    quantity: int
    added_at: datetime


class CartItemCreate(BaseSchema):
    product_id: int
    quantity: int = Field(1, ge=1)


class CartResponse(BaseSchema):
    user_id: int
    cart_items: List[Dict[str, Any]]
    total_items: int
    total_quantity: int
    total_price: float


class OrderBase(BaseSchema):
    order_id: int
    user_id: int
    total_price: float
    status: str
    order_date: datetime
    completed_at: Optional[datetime] = None
    items: Optional[Dict[str, Any]] = None


class OrderResponse(BaseSchema):
    message: str
    order: Dict[str, Any]


class UserSessionBase(BaseSchema):
    session_id: str
    user_id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    session_length: int
    total_reward: float
    actions_count: int
    experiment_id: Optional[str] = None
    session_number: int


class UserSessionCreate(BaseSchema):
    user_id: int
    experiment_id: Optional[str] = None

class UserSessionResponse(BaseSchema):
    session_id: str
    user_id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    session_length: int
    total_reward: float
    actions_count: int
    experiment_id: Optional[str] = None
    is_active: bool = Field(..., description="Whether the session is still active")
    session_number: int


# Batch training schemas
class BulkUserRegistration(BaseSchema):
    count: int = Field(..., ge=1, le=1000, description="Number of users to register (1-1000)")
    
class BulkUserSessionInfo(BaseSchema):
    user_id: int
    session_id: str
    session_number: int
    started_at: datetime


class BulkUserResponse(BaseSchema):
    message: str
    users_created: int
    user_ids: List[str]
    user_sessions: List[BulkUserSessionInfo]

class BatchAction(BaseSchema):
    user_id: int
    product_id: int
    action: str = Field(..., pattern="^(like|dislike|add_to_cart|report)$")

class BulkActionsRequest(BaseSchema):
    actions: List[BatchAction] = Field(..., max_items=10000)
    
class BulkActionsResponse(BaseSchema):
    message: str
    actions_processed: int
    learning_updates: int
    
class TrainingStatus(BaseSchema):
    total_users: int
    total_actions: int
    learning_episodes: int
    current_epsilon: float
    average_reward: float
    last_update: str
    
class SimulationRequest(BaseSchema):
    num_users: int = Field(..., ge=1, le=100, description="Number of users to simulate")
    actions_per_user: int = Field(..., ge=1, le=100, description="Actions per user")
    simulation_speed: float = Field(1.0, ge=0.1, le=10.0, description="Speed multiplier")
    
class SimulationResponse(BaseSchema):
    message: str
    simulation_id: str
    users_created: int
    total_actions: int
    estimated_duration: float

# Experiment management schemas
class ExperimentConfiguration(BaseSchema):
    name: str = Field(..., description="Experiment name")
    description: str = Field("", description="Experiment description")
    n_products: int = Field(500, ge=100, le=2000, description="Number of products")
    n_users: int = Field(100, ge=10, le=1000, description="Number of users to create")
    actions_per_user: int = Field(20, ge=1, le=100, description="Actions per user")
    simulation_speed: float = Field(2.0, ge=0.1, le=10.0, description="Simulation speed")
    agent_type: str = Field("dqn", pattern="^(dqn|epsilon_greedy|linucb|random)$")


class ExperimentResults(BaseSchema):
    experiment_id: str
    total_users: int
    total_actions: int
    total_rewards: float
    average_reward: float
    action_distribution: Dict[str, int]
    learning_curve: List[float]
    completion_time: float
    agent_performance: Dict[str, Any]
    reward_distribution: Dict[str, Dict[str, float]]
    conversion_metrics: Dict[str, float]
    session_metrics: Dict[str, Any]
    reward_timeline: List[Dict[str, float]]

class StartExperimentResponse(BaseSchema):
    message: str
    experiment_id: str
    status: str
    estimated_duration: float
