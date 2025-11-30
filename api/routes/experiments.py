"""
Experiment management routes for ML experiments with configurable parameters.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db
from api.models.schemas import (
    ExperimentConfiguration, ExperimentStatus, ExperimentResults,
    StartExperimentResponse
)
from api.services.experiment_service import ExperimentService

router = APIRouter(prefix="/experiments", tags=["experiments"])

@router.post("/start", response_model=StartExperimentResponse)
async def start_experiment(
    config: ExperimentConfiguration, 
    db: AsyncSession = Depends(get_db)
):
    """
    Start a new ML experiment with configurable parameters.

    - **name**: Experiment name
    - **description**: Optional description
    - **n_products**: Number of products (100-2000)
    - **n_users**: Number of users to create (10-1000)
    - **actions_per_user**: Actions per user (1-100)
    - **simulation_speed**: Speed multiplier (0.1-10.0)
    - **agent_type**: ML agent type (dqn, epsilon_greedy, linucb)

    Returns experiment ID and estimated duration.
    """
    try:
        experiment_service = ExperimentService()
        result = await experiment_service.create_experiment(db, config)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start experiment: {str(e)}")

@router.get("/", response_model=List[ExperimentStatus])
async def list_experiments(db: AsyncSession = Depends(get_db)):
    """
    List all experiments with their current status.

    Returns list of experiments with:
    - experiment_id, name, status, progress
    - current_agent, start_time, end_time, created_at
    - error message (if failed)
    """
    try:
        experiment_service = ExperimentService()
        experiments = await experiment_service.list_experiments(db)
        print(f"✅ Retrieved {len(experiments)} experiments from service")
        return experiments
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list experiments: {str(e)}")

@router.get("/{experiment_id}/status", response_model=ExperimentStatus)
async def get_experiment_status(
    experiment_id: str, 
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed status of a specific experiment.

    - **experiment_id**: Unique experiment identifier

    Returns current status, progress, and configuration.
    """
    try:
        experiment_service = ExperimentService()
        status = await experiment_service.get_experiment_status(db, experiment_id)
        if not status:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return status
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get experiment status: {str(e)}")

@router.get("/{experiment_id}/results", response_model=ExperimentResults)
async def get_experiment_results(
    experiment_id: str, 
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed results of a completed experiment.

    - **experiment_id**: Unique experiment identifier

    Returns comprehensive results including:
    - Performance metrics
    - Action distribution
    - Learning curve
    - Agent performance
    """
    try:
        experiment_service = ExperimentService()
        results = await experiment_service.get_experiment_results(db, experiment_id)
        if not results:
            # Check if experiment exists but not completed
            status = await experiment_service.get_experiment_status(db, experiment_id)
            if not status:
                raise HTTPException(
                    status_code=404, detail="Experiment not found")
            elif status.status != "completed":
                raise HTTPException(
                    status_code=400, detail=f"Experiment is {status.status}, results not available")
            else:
                raise HTTPException(
                    status_code=404, detail="Results not found")
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get experiment results: {str(e)}")

@router.post("/{experiment_id}/stop")
async def stop_experiment(
    experiment_id: str, 
    db: AsyncSession = Depends(get_db)
):
    """
    Stop a running experiment.

    - **experiment_id**: Unique experiment identifier

    Cancels the experiment if it's currently running.
    """
    try:
        experiment_service = ExperimentService()
        # First check if experiment exists
        status = await experiment_service.get_experiment_status(db, experiment_id)
        if not status:
            raise HTTPException(status_code=404, detail="Experiment not found")

        success = await experiment_service.stop_experiment(db, experiment_id)
        if not success:
            raise HTTPException(
                status_code=400, detail=f"Cannot stop experiment with status: {status.status}")

        return {"message": f"Experiment {experiment_id} stopped successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to stop experiment: {str(e)}")

@router.delete("/{experiment_id}")
async def delete_experiment(
    experiment_id: str, 
    db: AsyncSession = Depends(get_db)
):
    """
    Delete an experiment and all its data.

    - **experiment_id**: Unique experiment identifier

    Removes experiment from storage. If running, stops it first.
    """
    try:
        experiment_service = ExperimentService()
        # First check if experiment exists
        status = await experiment_service.get_experiment_status(db, experiment_id)
        if not status:
            raise HTTPException(status_code=404, detail="Experiment not found")
            
        success = await experiment_service.delete_experiment(db, experiment_id)
        if not success:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        return {"message": f"Experiment {experiment_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete experiment: {str(e)}")

@router.get("/templates/quick-test", response_model=ExperimentConfiguration)
async def get_quick_test_template():
    """
    Get a pre-configured template for quick testing.

    Returns a configuration suitable for rapid testing:
    - Small dataset (100 products, 20 users)
    - Fast simulation (5 actions per user, 5x speed)
    - DQN agent
    """
    return ExperimentConfiguration(
        name="Quick Test",
        description="Fast experiment for testing the system",
        n_products=100,
        n_users=20,
        actions_per_user=5,
        simulation_speed=5.0,
        agent_type="dqn"
    )

@router.get("/templates/full-experiment", response_model=ExperimentConfiguration)
async def get_full_experiment_template():
    """
    Get a pre-configured template for comprehensive experiments.

    Returns a configuration for thorough testing:
    - Large dataset (500 products, 100 users)
    - Comprehensive simulation (20 actions per user, 2x speed)
    - DQN agent
    """
    return ExperimentConfiguration(
        name="Full Experiment",
        description="Comprehensive experiment with full dataset",
        n_products=500,
        n_users=100,
        actions_per_user=20,
        simulation_speed=2.0,
        agent_type="dqn"
    )

@router.get("/templates/agent-comparison", response_model=List[ExperimentConfiguration])
async def get_agent_comparison_templates():
    """
    Get pre-configured templates for comparing different agents.

    Returns configurations for testing all agent types:
    - DQN, Epsilon-Greedy, LinUCB
    - Same parameters for fair comparison
    """
    base_config = {
        "description": "Agent comparison experiment",
        "n_products": 300,
        "n_users": 50,
        "actions_per_user": 15,
        "simulation_speed": 3.0
    }

    agents = ["dqn", "epsilon_greedy", "linucb"]
    templates = []

    for agent in agents:
        config = ExperimentConfiguration(
            name=f"Agent Comparison - {agent.upper()}",
            agent_type=agent,
            **base_config
        )
        templates.append(config)

    return templates