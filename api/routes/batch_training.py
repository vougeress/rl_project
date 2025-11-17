"""
Batch training routes for bulk user creation and actions simulation.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from api.models.schemas import (
    BulkUserRegistration, BulkUserResponse,
    BulkActionsRequest, BulkActionsResponse,
    TrainingStatus, SimulationRequest, SimulationResponse
)
from api.services.batch_training_service import BatchTrainingService
from api.core.database import get_db

router = APIRouter(prefix="/batch", tags=["batch-training"])

@router.post("/users/bulk-register", response_model=BulkUserResponse)
async def bulk_register_users(
    request: BulkUserRegistration,
    experiment_id: Optional[str] = Query(None, description="Optional experiment ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Register multiple users at once for training purposes.
    
    - **count**: Number of users to register (1-1000)
    - **experiment_id**: Optional experiment identifier
    
    Returns list of created user IDs.
    """
    try:
        batch_service = BatchTrainingService(db)
        result = await batch_service.bulk_register_users(request.count, experiment_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk registration failed: {str(e)}")

@router.post("/actions/bulk-process", response_model=BulkActionsResponse)
async def bulk_process_actions(
    request: BulkActionsRequest,
    experiment_id: Optional[str] = Query(None, description="Optional experiment ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Process multiple user actions at once for training.
    
    - **actions**: List of user actions (max 10,000)
    - **experiment_id**: Optional experiment identifier
    
    Each action should contain:
    - user_id: User ID
    - product_id: Product ID  
    - action: "like", "dislike", "add_to_cart", or "report"
    """
    try:
        # Convert Pydantic models to dictionaries
        actions_data = [action.model_dump() for action in request.actions]
        
        # Add experiment_id to each action if provided
        if experiment_id:
            for action in actions_data:
                action['experiment_id'] = experiment_id
        
        batch_service = BatchTrainingService(db)
        result = await batch_service.process_bulk_actions(actions_data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk actions processing failed: {str(e)}")

@router.post("/simulate", response_model=SimulationResponse)
async def simulate_user_behavior(
    request: SimulationRequest,
    experiment_id: Optional[str] = Query(None, description="Optional experiment ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Simulate realistic user behavior for training the recommendation system.
    
    - **num_users**: Number of users to simulate (1-100)
    - **actions_per_user**: Actions per user (1-100)
    - **simulation_speed**: Speed multiplier (0.1-10.0)
    - **experiment_id**: Optional experiment identifier
    
    This endpoint will:
    1. Create the specified number of users
    2. Generate realistic interactions for each user
    3. Process all actions through the learning system
    4. Return simulation statistics
    """
    try:
        batch_service = BatchTrainingService(db)
        result = await batch_service.simulate_user_behavior(
            request.num_users,
            request.actions_per_user,
            request.simulation_speed,
            experiment_id
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@router.get("/training/status", response_model=TrainingStatus)
async def get_training_status(db: AsyncSession = Depends(get_db)):
    """
    Get current training statistics and system status.
    
    Returns:
    - Total users created
    - Total actions processed
    - Learning episodes completed
    - Current agent parameters (epsilon, etc.)
    - Average reward
    - Last update timestamp
    """
    try:
        batch_service = BatchTrainingService(db)
        status = await batch_service.get_training_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {str(e)}")

@router.post("/training/reset")
async def reset_training(db: AsyncSession = Depends(get_db)):
    """
    Reset training statistics and optionally reset the learning agent.
    
    Use this to start fresh training sessions.
    """
    try:
        batch_service = BatchTrainingService(db)
        await batch_service.reset_training_stats()
        return {"message": "Training statistics reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset training: {str(e)}")

@router.post("/training/quick-demo")
async def quick_training_demo(
    experiment_id: Optional[str] = Query(None, description="Optional experiment ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    Quick demo: Create 10 users and simulate 50 actions for immediate training.
    
    - **experiment_id**: Optional experiment identifier
    
    This is a convenience endpoint for quick testing and demonstration.
    """
    try:
        batch_service = BatchTrainingService(db)
        result = await batch_service.simulate_user_behavior(
            num_users=10,
            actions_per_user=5,
            simulation_speed=2.0,
            experiment_id=experiment_id
        )
        return {
            "message": "Quick demo completed",
            "demo_results": result,
            "next_steps": "Check /batch/training/status for updated statistics"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick demo failed: {str(e)}")