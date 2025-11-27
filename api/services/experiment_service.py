"""
Service for managing ML experiments with configurable parameters.
"""

import asyncio
import random
import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from api.core.learning_manager import learning_manager, ensure_learning_system_ready
from api.services.batch_training_service import BatchTrainingService
from api.services.user_service import UserService
from api.models.database_models import Experiment, UserAction
from api.models.schemas import (
    ExperimentConfiguration, ExperimentStatus, ExperimentResults, 
    StartExperimentResponse
)
from src.data_generation import generate_synthetic_data
from src.agents.factory import create_agent


class ExperimentService:
    def __init__(self):
        self.running_experiments: Dict[str, asyncio.Task] = {}
        self.user_service = UserService()
    
    async def create_experiment(self, db: AsyncSession, config: ExperimentConfiguration) -> StartExperimentResponse:
        """Create and start a new experiment."""
        try:
            is_ready = await ensure_learning_system_ready()
            if not is_ready:
                raise RuntimeError("Learning system is not ready and cannot be initialized")
            
            experiment_id = str(uuid.uuid4())[:8]
            
            # Create experiment in database
            experiment = Experiment(
                experiment_id=experiment_id,
                name=config.name,
                description=config.description,
                status='pending',
                progress=0.0,
                configuration=config.model_dump(),
                agent_type=config.agent_type,
                agent_params=config.model_dump(),
                start_time=datetime.now()
            )
            
            db.add(experiment)
            await db.commit()
            
            # Start experiment in background
            task = asyncio.create_task(self._run_experiment(db, experiment_id, config))
            self.running_experiments[experiment_id] = task
            
            estimated_duration = (config.n_users * config.actions_per_user * 0.1) / config.simulation_speed
            
            return StartExperimentResponse(
                message=f"Experiment '{config.name}' started",
                experiment_id=experiment_id,
                status="running",
                estimated_duration=estimated_duration
            )
        except Exception as e:
            await db.rollback()
            raise e
    
    async def _run_experiment(self, db: AsyncSession, experiment_id: str, config: ExperimentConfiguration):
        """Run the experiment in background."""
        experiment = None
        try:
            # Update status to running
            stmt = select(Experiment).where(Experiment.experiment_id == experiment_id)
            result = await db.execute(stmt)
            experiment = result.scalar_one_or_none()
            
            if experiment:
                experiment.status = 'running'
                experiment.progress = 0.1
                await db.commit()
            
            start_time = time.time()
            
            # Step 1: Ensure system is ready (должна быть готова после create_experiment)
            if not learning_manager.is_ready:
                print(f"❌ Learning system not ready for experiment {experiment_id}, aborting")
                if experiment:
                    experiment.status = 'failed'
                    experiment.error_message = "Learning system not ready"
                    experiment.end_time = datetime.now()
                    await db.commit()
                return
            
            # Step 2: Create users and simulate behavior
            total_actions = 0
            total_rewards = 0.0
            action_counts = {}
            action_stats = defaultdict(lambda: {"count": 0, "total_reward": 0.0})
            learning_curve = []
            reward_timeline = []
            
            # Create users in batches
            users_created = 0
            batch_size = min(50, config.n_users)
            
            batch_service = BatchTrainingService(db)  # Передаем db в конструктор
            user_session_cache: Dict[int, Dict[str, Any]] = {}
            
            for batch_start in range(0, config.n_users, batch_size):
                batch_end = min(batch_start + batch_size, config.n_users)
                batch_count = batch_end - batch_start
                
                # Register users through batch service
                bulk_result = await batch_service.bulk_register_users(batch_count, experiment_id)
                registered_user_ids = [int(uid) for uid in bulk_result['user_ids']]
                users_created += batch_count
                for session_info in bulk_result.get('user_sessions', []):
                    user_session_cache[int(session_info['user_id'])] = {
                        'session_id': session_info['session_id'],
                        'session_number': session_info.get('session_number', 1)
                    }
                
                # Simulate actions for each user
                for user_id in registered_user_ids:
                    session_meta = user_session_cache.get(user_id)
                    if not session_meta:
                        session = await self.user_service.get_user_session(db, user_id)
                        if not session:
                            session = await self.user_service.start_new_session(db, user_id, experiment_id)
                        session_meta = {
                            'session_id': session.session_id,
                            'session_number': session.session_number
                        }
                        user_session_cache[user_id] = session_meta
                    session_id = session_meta['session_id']
                    for action_num in range(config.actions_per_user):
                        try:
                            # Get recommendations using the learning manager
                            recommendations = await learning_manager.get_recommendations(user_id, 20)
                            
                            if recommendations:
                                # Choose random product
                                product = random.choice(recommendations)
                                product_id = product['product_id']
                                
                                # Get product features for realistic simulation
                                product_features = learning_manager.catalog.get_product_features(product_id)
                                
                                # Simulate user interaction
                                simulator_user_id = user_id % learning_manager.simulator.n_users
                                interaction_result = learning_manager.simulator.simulate_user_interaction(
                                    simulator_user_id, product_features, action_num
                                )
                                
                                # Get total reward
                                total_step_reward = interaction_result['total_reward']
                                
                                # Process each occurred action and save to database
                                for occurred_action in interaction_result['occurred_actions']:
                                    action_name = occurred_action['action']
                                    action_reward = occurred_action['reward']
                                    
                                    # Save action to database
                                    user_action = UserAction(
                                        user_id=user_id,
                                        product_id=product_id,
                                        action_type=action_name,
                                        reward=action_reward,
                                        session_time=action_num,
                                        action_timestamp=datetime.now(),
                                        experiment_id=experiment_id,
                                        session_id=session_id
                                    )
                                    db.add(user_action)
                                    
                                    await self.user_service.update_session_on_action(
                                        db, session_id, action_reward, auto_commit=False
                                    )
                                    session_state = await self.user_service.update_user_state_vector(
                                        db, user_id, session_id, action_name, action_reward
                                    )
                                    
                                    session_context = {
                                        'current_session_actions': session_state.get('current_session_actions', 0),
                                        'current_session_reward': session_state.get('current_session_reward', 0.0),
                                        'average_reward': session_state.get('average_reward', action_reward)
                                    }
                                    
                                    # Update agent
                                    learning_manager.learn_from_action(
                                        user_id,
                                        product_id,
                                        action_name,
                                        action_reward,
                                        session_context=session_context
                                    )
                                    
                                    # Track statistics
                                    action_counts[action_name] = action_counts.get(action_name, 0) + 1
                                    action_stats[action_name]["count"] += 1
                                    action_stats[action_name]["total_reward"] += action_reward
                                
                                # Track overall statistics
                                total_actions += 1
                                total_rewards += total_step_reward
                                
                                # Record learning curve
                                if total_actions % 100 == 0:
                                    avg_reward = total_rewards / total_actions
                                    learning_curve.append(avg_reward)
                                    reward_timeline.append({
                                        "actions": total_actions,
                                        "avg_reward": avg_reward
                                    })
                                
                                # Small delay for simulation speed
                                await asyncio.sleep(0.01 / config.simulation_speed)
                        
                        except Exception as e:
                            print(f"Error in experiment {experiment_id} for user {user_id}: {e}")
                            continue
                
                    await self.user_service.end_session(db, session_id, auto_commit=False)
                    new_session = await self.user_service.start_new_session(db, user_id, experiment_id)
                    user_session_cache[user_id] = {
                        'session_id': new_session.session_id,
                        'session_number': new_session.session_number
                    }
                
                # Update progress
                if experiment:
                    progress = 0.5 + 0.4 * (users_created / config.n_users)
                    experiment.progress = progress
                    await db.commit()
            
            # Step 4: Calculate final results and save to database
            completion_time = time.time() - start_time
            average_reward = total_rewards / max(total_actions, 1)
            
            reward_distribution = {}
            for action_name, stats in action_stats.items():
                count = stats["count"]
                reward_distribution[action_name] = {
                    "count": count,
                    "avg_reward": stats["total_reward"] / max(count, 1),
                    "percentage": count / max(total_actions, 1)
                }
            if total_actions > 0:
                reward_timeline.append({
                    "actions": total_actions,
                    "avg_reward": average_reward
                })
            
            def get_rate(action: str) -> float:
                return action_counts.get(action, 0) / max(total_actions, 1)
            
            conversion_metrics = {
                "view_rate": get_rate('view'),
                "interaction_rate": (action_counts.get('like', 0) + action_counts.get('share', 0)) / max(total_actions, 1),
                "cart_rate": get_rate('add_to_cart'),
                "purchase_rate": get_rate('purchase'),
                "negative_feedback_rate": (
                    action_counts.get('dislike', 0) +
                    action_counts.get('report', 0) +
                    action_counts.get('report_spam', 0) +
                    action_counts.get('close_immediately', 0)
                ) / max(total_actions, 1)
            }
            
            session_metrics = {
                "sessions": users_created,
                "avg_actions_per_session": total_actions / max(users_created, 1),
                "avg_reward_per_session": total_rewards / max(users_created, 1),
                "configured_actions_per_user": config.actions_per_user,
                "completion_time_per_session": completion_time / max(users_created, 1)
            }
            
            # Agent performance metrics
            agent_performance = {
                'agent_type': config.agent_type,
                'total_episodes': len(learning_manager.learning_history),
                'final_epsilon': getattr(learning_manager.agent, 'epsilon', 0.0) if hasattr(learning_manager.agent, 'epsilon') else 0.0,
                'learning_rate': getattr(learning_manager.agent, 'learning_rate', 0.0) if hasattr(learning_manager.agent, 'learning_rate') else 0.0
            }
            
            # Store results in database
            if experiment:
                experiment.status = 'completed'
                experiment.progress = 1.0
                experiment.end_time = datetime.now()
                experiment.results = {
                    'total_users': users_created,
                    'total_actions': total_actions,
                    'total_rewards': total_rewards,
                    'average_reward': average_reward,
                    'action_distribution': action_counts,
                    'learning_curve': learning_curve,
                    'completion_time': completion_time,
                    'agent_performance': agent_performance,
                    'reward_distribution': reward_distribution,
                    'conversion_metrics': conversion_metrics,
                    'session_metrics': session_metrics,
                    'reward_timeline': reward_timeline
                }
                await db.commit()
                
        except Exception as e:
            # Handle experiment failure
            if experiment:
                experiment.status = 'failed'
                experiment.error_message = str(e)
                experiment.end_time = datetime.now()
                await db.commit()
            print(f"Experiment {experiment_id} failed: {e}")
        
        finally:
            # Clean up running task
            if experiment_id in self.running_experiments:
                del self.running_experiments[experiment_id]
    
    async def get_experiment_status(self, db: AsyncSession, experiment_id: str) -> Optional[ExperimentStatus]:
        """Get experiment status by ID."""
        try:
            stmt = select(Experiment).where(Experiment.experiment_id == experiment_id)
            result = await db.execute(stmt)
            experiment = result.scalar_one_or_none()
            
            if not experiment:
                return None
                
            return ExperimentStatus(
                experiment_id=experiment.experiment_id,
                name=experiment.name,
                description=experiment.description,
                status=experiment.status,
                progress=experiment.progress,
                current_agent=experiment.agent_type,
                start_time=experiment.start_time,
                end_time=experiment.end_time,
                error=experiment.error_message,
                created_at=experiment.created_at
            )
        except Exception as e:
            raise ValueError(f"Failed to get experiment status: {str(e)}")
    
    async def get_experiment_results(self, db: AsyncSession, experiment_id: str) -> Optional[ExperimentResults]:
        """Get experiment results by ID."""
        try:
            stmt = select(Experiment).where(Experiment.experiment_id == experiment_id)
            result = await db.execute(stmt)
            experiment = result.scalar_one_or_none()
            
            if not experiment or not experiment.results:
                return None
            
            results_data = experiment.results
            return ExperimentResults(
                experiment_id=experiment_id,
                total_users=results_data.get('total_users', 0),
                total_actions=results_data.get('total_actions', 0),
                total_rewards=results_data.get('total_rewards', 0.0),
                average_reward=results_data.get('average_reward', 0.0),
                action_distribution=results_data.get('action_distribution', {}),
                learning_curve=results_data.get('learning_curve', []),
                completion_time=results_data.get('completion_time', 0.0),
                agent_performance=results_data.get('agent_performance', {}),
                reward_distribution=results_data.get('reward_distribution', {}),
                conversion_metrics=results_data.get('conversion_metrics', {}),
                session_metrics=results_data.get('session_metrics', {}),
                reward_timeline=results_data.get('reward_timeline', [])
            )
        except Exception as e:
            raise ValueError(f"Failed to get experiment results: {str(e)}")
    
    async def list_experiments(self, db: AsyncSession) -> List[ExperimentStatus]:
        """List all experiments."""
        try:
            stmt = select(Experiment).order_by(Experiment.created_at.desc())
            result = await db.execute(stmt)
            experiments = result.scalars().all()

            print(f"✅ Found {len(experiments)} experiments in database")
            
            return [
                ExperimentStatus(
                    experiment_id=exp.experiment_id,
                    name=exp.name,
                    description=exp.description,
                    agent_type=exp.agent_type,
                    status=exp.status,
                    progress=exp.progress,
                    configuration=exp.configuration,
                    agent_params=exp.agent_params,
                    results=exp.results,
                    start_time=exp.start_time,
                    end_time=exp.end_time,
                    created_at=exp.created_at,
                    error=exp.error_message
                )
                for exp in experiments
            ]
        except Exception as e:
            print(f"❌ Error in list_experiments service method: {e}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Failed to list experiments: {str(e)}")
    
    async def stop_experiment(self, db: AsyncSession, experiment_id: str) -> bool:
        """Stop a running experiment."""
        try:
            if experiment_id in self.running_experiments:
                task = self.running_experiments[experiment_id]
                task.cancel()
                
                # Update status in database
                stmt = select(Experiment).where(Experiment.experiment_id == experiment_id)
                result = await db.execute(stmt)
                experiment = result.scalar_one_or_none()
                
                if experiment:
                    experiment.status = 'cancelled'
                    experiment.end_time = datetime.now()
                    await db.commit()
                
                del self.running_experiments[experiment_id]
                return True
            return False
        except Exception as e:
            await db.rollback()
            raise ValueError(f"Failed to stop experiment: {str(e)}")
    
    async def delete_experiment(self, db: AsyncSession, experiment_id: str) -> bool:
        """Delete an experiment and its results."""
        try:
            # Stop if running
            if experiment_id in self.running_experiments:
                await self.stop_experiment(db, experiment_id)
            
            # Delete from database
            stmt = select(Experiment).where(Experiment.experiment_id == experiment_id)
            result = await db.execute(stmt)
            experiment = result.scalar_one_or_none()
            
            if experiment:
                await db.delete(experiment)
                await db.commit()
                return True
            return False
        except Exception as e:
            await db.rollback()
            raise ValueError(f"Failed to delete experiment: {str(e)}")
