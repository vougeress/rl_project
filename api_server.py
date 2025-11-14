"""
FastAPI server for E-commerce RL Recommendation System.
Reorganized with proper folder structure.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import new organized API components
from api.routes.recommendations import router as recommendations_router
from api.routes.users import router as users_router
from api.routes.cart import router as cart_router
from api.routes.batch_training import router as batch_training_router
from api.routes.experiments import router as experiments_router

# FastAPI app
app = FastAPI(
    title="E-commerce RL Recommendation API",
    description="REST API for reinforcement learning recommendation system",
    version="2.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(recommendations_router)
app.include_router(users_router)
app.include_router(cart_router)
app.include_router(batch_training_router)
app.include_router(experiments_router)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "E-commerce RL Recommendation API",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Smart recommendations with DQN agent",
            "Continuous learning from user actions",
            "User registration with dataset medians",
            "Shopping cart and order placement",
            "Batch training with bulk user creation",
            "User behavior simulation for ML training",
            "Real-time training statistics monitoring",
            "ML experiment management with configurable parameters",
            "Multi-agent comparison experiments",
            "Real-time experiment monitoring and results visualization"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)