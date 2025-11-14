"""
Service for managing ML experiments with configurable parameters.
"""

import asyncio
import uuid
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from api.core.learning_manager import GlobalLearningManager
from api.services.batch_training_service import BatchTrainingService
from api.models.schemas import (
    ExperimentConfiguration, ExperimentStatus, ExperimentResults, 
    StartExperimentResponse
)
from src.data_generation import generate_synthetic_data
from src.agents.factory import create_agent


class ExperimentService:
    def __init__(self):
        self.experiments: Dict[str, ExperimentStatus] = {}
        self.experiment_results: Dict[str, ExperimentResults] = {}
        self.batch_service = BatchTrainingService()
        self.running_experiments: Dict[str, asyncio.Task] = {}
    
    async def create_experiment(self, config: ExperimentConfiguration) -> StartExperimentResponse:
        """Create and start a new experiment."""
        experiment_id = str(uuid.uuid4())[:8]
        
        # Create experiment status
        experiment_status = ExperimentStatus(
            experiment_id=experiment_id,
            name=config.name,
            status="pending",
            progress=0.0,
            configuration=config
        )
        
        self.experiments[experiment_id] = experiment_status
        
        # Start experiment in background
        task = asyncio.create_task(self._run_experiment(experiment_id, config))
        self.running_experiments[experiment_id] = task
        
        estimated_duration = (config.n_users * config.actions_per_user * 0.1) / config.simulation_speed
        
        return StartExperimentResponse(
            message=f"Experiment '{config.name}' started",
            experiment_id=experiment_id,
            status="running",
            estimated_duration=estimated_duration
        )
    
    async def _run_experiment(self, experiment_id: str, config: ExperimentConfiguration):
        """Run the experiment in background."""
        try:
            # Update status to running
            self.experiments[experiment_id].status = "running"
            self.experiments[experiment_id].start_time = datetime.now().isoformat()
            self.experiments[experiment_id].progress = 0.1
            
            start_time = time.time()
            
            # Step 1: Initialize system with custom parameters
            learning_manager = GlobalLearningManager()
            catalog, simulator = generate_synthetic_data(
                config.n_products, 
                config.n_users
            )
            
            # Create custom agent
            state_dim = len(simulator.get_user_state(0))
            agent = create_agent(config.agent_type, config.n_products, state_dim)
            
            # Replace the global agent
            learning_manager.catalog = catalog
            learning_manager.simulator = simulator
            learning_manager.agent = agent
            learning_manager.is_ready = True
            
            self.experiments[experiment_id].progress = 0.3
            
            # Step 2: Pre-train agent
            await self._pretrain_agent(learning_manager, experiment_id)
            self.experiments[experiment_id].progress = 0.5
            
            # Step 3: Create users and simulate behavior
            total_actions = 0
            total_rewards = 0.0
            action_counts = {}
            learning_curve = []
            
            # Create users in batches
            users_created = 0
            batch_size = min(50, config.n_users)
            
            for batch_start in range(0, config.n_users, batch_size):
                batch_end = min(batch_start + batch_size, config.n_users)
                batch_count = batch_end - batch_start
                
                # Register users
                bulk_result = await self.batch_service.bulk_register_users(batch_count)
                registered_user_ids = bulk_result['user_ids']
                users_created += batch_count
                
                # Simulate actions for this batch - use simulator user IDs (0-based)
                for i, registered_user_id in enumerate(registered_user_ids):
                    # Map to simulator user ID (within bounds)
                    simulator_user_id = (batch_start + i) % learning_manager.simulator.n_users
                    for action_num in range(config.actions_per_user):
                        try:
                            # Get recommendations using registered user ID
                            recommendations = learning_manager.get_recommendations(int(registered_user_id), 20)
                            
                            if recommendations:
                                # Choose random product
                                import random
                                product = random.choice(recommendations)
                                product_id = product['product_id']
                                
                                # Get product features for realistic simulation
                                product_features = learning_manager.catalog.get_product_features(product_id)
                                
                                # Use realistic user interaction simulation with simulator user ID
                                interaction_result = learning_manager.simulator.simulate_user_interaction(
                                    simulator_user_id, product_features, action_num
                                )
                                
                                # Get total reward from all occurred actions
                                total_step_reward = interaction_result['total_reward']
                                
                                # Process each occurred action
                                for occurred_action in interaction_result['occurred_actions']:
                                    action_name = occurred_action['action']
                                    action_reward = occurred_action['reward']
                                    
                                    # Update agent with this specific action (use registered user ID for learning)
                                    learning_manager.learn_from_action(int(registered_user_id), product_id, action_name, action_reward)
                                    
                                    # Track statistics
                                    action_counts[action_name] = action_counts.get(action_name, 0) + 1
                                
                                # Track overall statistics
                                total_actions += 1
                                total_rewards += total_step_reward
                                
                                # Record learning curve
                                if total_actions % 100 == 0:
                                    avg_reward = total_rewards / total_actions
                                    learning_curve.append(avg_reward)
                                
                                # Small delay for simulation speed
                                await asyncio.sleep(0.01 / config.simulation_speed)
                        
                        except Exception as e:
                            print(f"Error in experiment {experiment_id}: {e}")
                            continue
                
                # Update progress
                progress = 0.5 + 0.4 * (users_created / config.n_users)
                self.experiments[experiment_id].progress = progress
            
            # Step 4: Calculate final results
            completion_time = time.time() - start_time
            average_reward = total_rewards / max(total_actions, 1)
            
            # Agent performance metrics
            agent_performance = {
                'agent_type': config.agent_type,
                'total_episodes': len(learning_manager.learning_history),
                'final_epsilon': getattr(agent, 'epsilon', 0.0) if hasattr(agent, 'epsilon') else 0.0,
                'learning_rate': getattr(agent, 'learning_rate', 0.0) if hasattr(agent, 'learning_rate') else 0.0
            }
            
            # Store results
            results = ExperimentResults(
                experiment_id=experiment_id,
                total_users=users_created,
                total_actions=total_actions,
                total_rewards=total_rewards,
                average_reward=average_reward,
                action_distribution=action_counts,
                learning_curve=learning_curve,
                completion_time=completion_time,
                agent_performance=agent_performance
            )
            
            self.experiment_results[experiment_id] = results
            
            # Update final status
            self.experiments[experiment_id].status = "completed"
            self.experiments[experiment_id].progress = 1.0
            self.experiments[experiment_id].end_time = datetime.now().isoformat()
            self.experiments[experiment_id].results = results.dict()
            
        except Exception as e:
            # Handle experiment failure
            self.experiments[experiment_id].status = "failed"
            self.experiments[experiment_id].error = str(e)
            self.experiments[experiment_id].end_time = datetime.now().isoformat()
            print(f"Experiment {experiment_id} failed: {e}")
        
        finally:
            # Clean up running task
            if experiment_id in self.running_experiments:
                del self.running_experiments[experiment_id]
    
    async def _pretrain_agent(self, learning_manager: GlobalLearningManager, experiment_id: str):
        """Pre-train the agent with realistic episodes."""
        episodes = 20
        
        for episode in range(episodes):
            # Use random user for each episode
            import random
            user_id = random.randint(0, learning_manager.simulator.n_users - 1)
            state = learning_manager.simulator.get_user_state(user_id)
            episode_reward = 0
            
            for step in range(15):
                action = learning_manager.agent.select_action(state)
                
                # Ensure action is valid
                if action >= learning_manager.catalog.n_products:
                    action = action % learning_manager.catalog.n_products
                
                # Get realistic reward using product features
                product_features = learning_manager.catalog.get_product_features(action)
                interaction_result = learning_manager.simulator.simulate_user_interaction(
                    user_id, product_features, step
                )
                
                # Use total reward from interaction
                reward = interaction_result['total_reward']
                next_state = learning_manager.simulator.get_user_state(user_id, step + 1)
                done = step >= 14
                
                # Train agent
                if hasattr(learning_manager.agent, 'update'):
                    learning_manager.agent.update(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Record learning
            learning_manager.learning_history.append({
                "episode": episode,
                "reward": episode_reward,
                "type": "pretrain",
                "timestamp": datetime.now()
            })
    
    def get_experiment_status(self, experiment_id: str) -> Optional[ExperimentStatus]:
        """Get experiment status by ID."""
        return self.experiments.get(experiment_id)
    
    def get_experiment_results(self, experiment_id: str) -> Optional[ExperimentResults]:
        """Get experiment results by ID."""
        return self.experiment_results.get(experiment_id)
    
    def list_experiments(self) -> List[ExperimentStatus]:
        """List all experiments."""
        return list(self.experiments.values())
    
    async def stop_experiment(self, experiment_id: str) -> bool:
        """Stop a running experiment."""
        if experiment_id in self.running_experiments:
            task = self.running_experiments[experiment_id]
            task.cancel()
            
            # Update status
            if experiment_id in self.experiments:
                self.experiments[experiment_id].status = "cancelled"
                self.experiments[experiment_id].end_time = datetime.now().isoformat()
            
            return True
        return False
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and its results."""
        # Stop if running
        if experiment_id in self.running_experiments:
            asyncio.create_task(self.stop_experiment(experiment_id))
        
        # Remove from storage
        removed = False
        if experiment_id in self.experiments:
            del self.experiments[experiment_id]
            removed = True
        
        if experiment_id in self.experiment_results:
            del self.experiment_results[experiment_id]
            removed = True
        
        return removed