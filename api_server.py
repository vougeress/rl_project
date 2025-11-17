"""
FastAPI server for E-commerce RL Recommendation System.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routes.recommendations import router as recommendations_router
from api.routes.users import router as users_router
from api.routes.cart import router as cart_router
from api.routes.batch_training import router as batch_training_router
from api.routes.experiments import router as experiments_router

from api.core.database import init_db
from api.core.learning_manager import learning_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting API...")
    
    print("🗄️ Initializing database...")
    await init_db()
    
    try:
        await learning_manager.initialize_system()
    except Exception as e:
        print(f"❌ Learning system initialization error: {e}")
    
    yield
    
    print("🛑 Shutting down API...")
    await learning_manager.shutdown()

# FastAPI app
app = FastAPI(
    title="E-commerce RL Recommendation API",
    description="REST API for reinforcement learning recommendation system",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(recommendations_router)
app.include_router(users_router)
app.include_router(cart_router)
app.include_router(batch_training_router)
app.include_router(experiments_router)

@app.get("/")
async def root():
    """Health check."""
    return {
        "message": "E-commerce RL Recommendation API",
        "version": "2.0.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    """Health check with learning system status."""
    learning_stats = learning_manager.get_learning_stats()
    return {
        "status": "healthy",
        "learning_system": learning_stats,
        "learning_system_ready": learning_manager.is_ready
    }

@app.post("/system/reinitialize")
async def reinitialize_system():
    """Reinitialize the learning system manually"""
    try:
        await learning_manager.shutdown()
        await learning_manager.initialize_system()
        return {
            "success": True,
            "ready": learning_manager.is_ready,
            "message": "System reinitialized successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reinitialization failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)