"""
Batch training routes for bulk user creation and actions simulation.
"""

from fastapi import APIRouter, HTTPException
from api.models.schemas import (
    BulkUserRegistration, BulkUserResponse,
    BulkActionsRequest, BulkActionsResponse,
    TrainingStatus, SimulationRequest, SimulationResponse
)
from api.services.batch_training_service import BatchTrainingService

router = APIRouter(prefix="/batch", tags=["batch-training"])
batch_service = BatchTrainingService()


@router.post("/users/bulk-register", response_model=BulkUserResponse)
async def bulk_register_users(request: BulkUserRegistration):
    """
    Register multiple users at once for training purposes.
    
    - **count**: Number of users to register (1-1000)
    
    Returns list of created user IDs.
    """
    try:
        result = await batch_service.bulk_register_users(request.count)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk registration failed: {str(e)}")


@router.post("/actions/bulk-process", response_model=BulkActionsResponse)
async def bulk_process_actions(request: BulkActionsRequest):
    """
    Process multiple user actions at once for training.
    
    - **actions**: List of user actions (max 10,000)
    
    Each action should contain:
    - user_id: User ID
    - product_id: Product ID  
    - action: "like", "dislike", "add_to_cart", or "report"
    """
    try:
        actions_data = [action.dict() for action in request.actions]
        result = await batch_service.process_bulk_actions(actions_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk actions processing failed: {str(e)}")


@router.post("/simulate", response_model=SimulationResponse)
async def simulate_user_behavior(request: SimulationRequest):
    """
    Simulate realistic user behavior for training the recommendation system.
    
    - **num_users**: Number of users to simulate (1-100)
    - **actions_per_user**: Actions per user (1-100)
    - **simulation_speed**: Speed multiplier (0.1-10.0)
    
    This endpoint will:
    1. Create the specified number of users
    2. Generate realistic interactions for each user
    3. Process all actions through the learning system
    4. Return simulation statistics
    """
    try:
        result = await batch_service.simulate_user_behavior(
            request.num_users,
            request.actions_per_user,
            request.simulation_speed
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@router.get("/training/status", response_model=TrainingStatus)
async def get_training_status():
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
        status = batch_service.get_training_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {str(e)}")


@router.post("/training/reset")
async def reset_training():
    """
    Reset training statistics and optionally reset the learning agent.
    
    Use this to start fresh training sessions.
    """
    try:
        await batch_service.reset_training_stats()
        return {"message": "Training statistics reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset training: {str(e)}")


@router.get("/training/quick-demo")
async def quick_training_demo():
    """
    Quick demo: Create 10 users and simulate 50 actions for immediate training.
    
    This is a convenience endpoint for quick testing and demonstration.
    """
    try:
        result = await batch_service.simulate_user_behavior(
            num_users=10,
            actions_per_user=5,
            simulation_speed=2.0
        )
        return {
            "message": "Quick demo completed",
            "demo_results": result,
            "next_steps": "Check /batch/training/status for updated statistics"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick demo failed: {str(e)}")