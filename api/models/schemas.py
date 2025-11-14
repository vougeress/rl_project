"""
Pydantic models for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class ExperimentConfig(BaseModel):
    n_products: int = 500
    n_users: int = 50
    n_episodes: int = 200
    max_session_length: int = 20
    agents: List[str] = ["epsilon_greedy", "linucb", "dqn", "random"]
    agent_params: Dict[str, Dict] = {}


class ExperimentStatus(BaseModel):
    experiment_id: str
    status: str  # "running", "completed", "failed"
    progress: float  # 0.0 to 1.0
    current_agent: Optional[str] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    error: Optional[str] = None


class RecommendationRequest(BaseModel):
    user_id: int
    session_time: int = 0
    agent_type: str = "dqn"
    experiment_id: Optional[str] = None


class UserActionResult(BaseModel):
    action: str
    reward: float
    description: str


class InteractionResult(BaseModel):
    occurred_actions: List[UserActionResult]
    total_reward: float
    action_probabilities: Dict[str, float]


class ProductInfo(BaseModel):
    product_id: int
    category: str
    price: float
    popularity: float
    quality: float


class UserInfo(BaseModel):
    user_id: int
    age: int
    income_level: str
    price_sensitivity: float
    quality_sensitivity: float
    exploration_tendency: float


class UserRegistration(BaseModel):
    name: str
    age: int


class UserAction(BaseModel):
    user_id: int
    product_id: int
    action: str  # "like", "dislike", "report", "add_to_cart"


class CartItem(BaseModel):
    product_id: int
    quantity: int = 1


class RecommendationsResponse(BaseModel):
    products: List[ProductInfo]
    user_id: int
    total_count: int


class UserRegistrationResponse(BaseModel):
    user_id: int
    name: str
    age: int
    income_level: str
    profile_completed_with_medians: Dict[str, float]
    message: str


class ProductActionResponse(BaseModel):
    user_id: int
    product_id: int
    product_info: ProductInfo
    action: str
    message: str
    reward: float
    timestamp: datetime


class CartResponse(BaseModel):
    user_id: int
    cart_items: List[Dict[str, Any]]
    total_items: int
    total_quantity: int
    total_price: float


class OrderResponse(BaseModel):
    message: str
    order: Dict[str, Any]


# Batch training schemas
class BulkUserRegistration(BaseModel):
    count: int = Field(..., ge=1, le=1000, description="Number of users to register (1-1000)")
    
class BulkUserResponse(BaseModel):
    message: str
    users_created: int
    user_ids: List[str]

class BatchAction(BaseModel):
    user_id: int
    product_id: int
    action: str = Field(..., pattern="^(like|dislike|add_to_cart|report)$")

class BulkActionsRequest(BaseModel):
    actions: List[BatchAction] = Field(..., max_items=10000)
    
class BulkActionsResponse(BaseModel):
    message: str
    actions_processed: int
    learning_updates: int
    
class TrainingStatus(BaseModel):
    total_users: int
    total_actions: int
    learning_episodes: int
    current_epsilon: float
    average_reward: float
    last_update: str
    
class SimulationRequest(BaseModel):
    num_users: int = Field(..., ge=1, le=100, description="Number of users to simulate")
    actions_per_user: int = Field(..., ge=1, le=100, description="Actions per user")
    simulation_speed: float = Field(1.0, ge=0.1, le=10.0, description="Speed multiplier")
    
class SimulationResponse(BaseModel):
    message: str
    simulation_id: str
    users_created: int
    total_actions: int
    estimated_duration: float

# Experiment management schemas
class ExperimentConfiguration(BaseModel):
    name: str = Field(..., description="Experiment name")
    description: str = Field("", description="Experiment description")
    n_products: int = Field(500, ge=100, le=2000, description="Number of products")
    n_users: int = Field(100, ge=10, le=1000, description="Number of users to create")
    actions_per_user: int = Field(20, ge=1, le=100, description="Actions per user")
    simulation_speed: float = Field(2.0, ge=0.1, le=10.0, description="Simulation speed")
    agent_type: str = Field("dqn", pattern="^(dqn|epsilon_greedy|linucb|random)$")
    
class ExperimentStatus(BaseModel):
    experiment_id: str
    name: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float  # 0.0 to 1.0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    configuration: ExperimentConfiguration
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ExperimentResults(BaseModel):
    experiment_id: str
    total_users: int
    total_actions: int
    total_rewards: float
    average_reward: float
    action_distribution: Dict[str, int]
    learning_curve: List[float]
    completion_time: float
    agent_performance: Dict[str, Any]

class StartExperimentResponse(BaseModel):
    message: str
    experiment_id: str
    status: str
    estimated_duration: float