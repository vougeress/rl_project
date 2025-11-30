# RL E-commerce Recommendation System

An end-to-end playground for reinforcement-learning driven recommendations in an e-commerce setting. The project couples a FastAPI backend, PostgreSQL storage, and a Streamlit dashboard to simulate shopper behaviour, train multiple agents (DQN, LinUCB, epsilon-greedy), and visualise experiment outcomes.

## Features

- **Synthetic shop simulation** – realistic catalogue/user generation (`src/data_generation.py`) and environment dynamics (`src/environment.py`).
- **Multiple RL agents** – modular agent factory (`src/agents/`) with DQN, LinUCB, and epsilon-greedy.
- **FastAPI service layer** – routes for recommendations, experiments, batch training, carts/orders, and user management (`api/routes/`).
- **Experiment manager** – asynchronous orchestration that spins up experiments, records user sessions, and computes rich metrics (`api/services/experiment_service.py`).
- **Analytics dashboard** – Streamlit UI (`unified_dashboard.py`) for launching comparisons and inspecting conversion funnels, reward distributions, session metrics, and learning curves.
- **Containerised stack** – Docker Compose orchestrates API, dashboard, PostgreSQL, and pgAdmin with persistent volumes.

## Simulation & Environment Details

The simulator stitches together products, users, and an environment that emits multi-action trajectories:

- **Products (`ProductCatalog`)**  
  - Generates up to 2,000 SKUs across 10 lifestyle categories with realistic priors.  
  - Each item carries a style vector (5 dimensions), quality, popularity, and price computed via log-normal distributions.  
  - Products are persisted to Postgres so recommendations can blend online learning with catalog metadata.

- **Users (`UserSimulator`)**  
  - Profiles sample age/income distributions, budget multipliers, category preferences (Dirichlet), and style affinities.  
  - Behavioural parameters (price/quality sensitivity, exploration tendency) steer both the RL environment and reward shaping.

- **Environment (`ECommerceEnv`)**  
  - Exposes a vectorised state: concatenation of user preference embeddings, product features, and interaction context.  
  - Supports multi-action outcomes per step (view, like, add_to_cart, purchase, share, dislike, report, report_spam, remove_from_cart).  
  - Rewards blend intrinsic action value, product quality bonus, and popularity bonus, enabling agents to trade off speed vs. revenue.  
  - The simulator injects stochasticity (epsilon-greedy sampling, time decay) so policies must handle shifting user preferences.

- **Session-aware logging**  
  - Every action is bound to a session id (`user_sessions` table) with cumulative reward, action counts, and session_length.  
  - Experiment runner records per-action stats, conversion funnel metrics, reward timelines, and learning curves for downstream analytics.

## Agent Implementations

All agents live in `src/agents/` and are instantiated via `src/agents/factory.py`:

| Agent | Key Files | Notes |
| --- | --- | --- |
| **Deep Q-Network (DQN)** | `src/agents/dqn.py` | Torch-based network with replay buffer, target network sync, epsilon decay. Input dimension equals environment state; output dimension equals catalog size (actions). |
| **LinUCB** | `src/agents/linucb.py` | Contextual bandit estimator with per-action covariance matrices; supports dynamic exploration via alpha parameter. |
| **Epsilon-Greedy** | `src/agents/epsilon_greedy.py` | Lightweight bandit maintaining running averages; decays epsilon, making it useful as a fast baseline. |

During experiments, the global learning manager (`api/core/learning_manager.py`) initialises the catalog, simulator, and default DQN agent, then exposes:

- `get_recommendations(user_id, limit)` – returns either agent-generated or popularity fallback lists.
- `learn_from_action(user, product, action, reward, session_context)` – updates agent weights and logs learning history.
- `update_user_preferences` – adapts per-user category/style preferences using recent windowed actions.

Each experiment can spawn any agent type; for non-DQN agents, the factory swaps the learning manager’s policy accordingly.

## Project Structure

```
.
├── api/                    # FastAPI application modules
├── src/                    # RL environment, agents, and simulators
├── unified_dashboard.py    # Streamlit dashboard entry point
├── api_server.py           # FastAPI bootstrap (uvicorn)
├── docker-compose.yml      # Multi-service stack
├── Dockerfile              # Base image for api/dashboard
├── postgres/               # DB init scripts
└── requirements.txt        # Python dependencies
```

## Getting Started

### Prerequisites

- Docker + Docker Compose
- Python 3.10+ (optional, for local execution without containers)

### Configuration

Create a `.env` file at the repository root (Docker Compose loads it automatically):

```
POSTGRES_DB=rl_db
POSTGRES_USER=rl_user
POSTGRES_PASSWORD=rl_password
API_PORT=8000
DATABASE_URL=postgresql://rl_user:rl_password@postgres:5432/rl_db
STREAMLIT_PORT=8501
PGADMIN_DEFAULT_EMAIL=admin@example.com
PGADMIN_DEFAULT_PASSWORD=adminpass
PGADMIN_PORT=5050
```

Adjust ports or credentials as needed. If you changed the host-exposed Postgres port (`docker-compose.yml` maps `5433:5432`), keep connecting via `localhost:5433`.

### Running with Docker

```bash
# Build images (first run or after dependency changes)
docker compose build

# Start API, dashboard, Postgres, pgAdmin
docker compose up -d

# Follow logs (optional)
docker compose logs -f api
```

Services after a successful start:

- FastAPI: http://localhost:8000 (docs at `/docs`)
- Streamlit dashboard: http://localhost:8501
- PostgreSQL: `localhost:5433` (use `.env` credentials)
- pgAdmin: http://localhost:5050

### Local Python Execution (optional)

1. Create a virtualenv and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Ensure PostgreSQL is running (Docker or local instance) and reachable via `DATABASE_URL`.
3. Launch the API:
   ```bash
   uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
   ```
4. Launch the dashboard:
   ```bash
   streamlit run unified_dashboard.py
   ```

## API Overview

All endpoints are defined under `api/routes/`. Highlights:

| Route | Description |
| --- | --- |
| `POST /users/register` | Register synthetic users. |
| `GET /recommendations/{user_id}` | Retrieve personalised product recommendations. |
| `POST /recommendations/{user_id}/action` | Submit user actions for learning updates. |
| `POST /experiments/start` | Launch a configurable experiment (agent, users, products, etc.). |
| `GET /experiments/` | List experiment statuses and metrics. |
| `GET /experiments/{id}/results` | Detailed metrics for a completed run. |
| `POST /batch/users/bulk-register` | Register multiple users at once. |
| `POST /batch/actions/bulk-process` | Process synthetic action batches for training. |
| `POST /batch/simulate` | Run a full simulation (user generation + actions). |
| `GET /cart/{user_id}` / `POST /cart/{user_id}` | Shopping cart operations and order placement. |

Use the OpenAPI docs at http://localhost:8000/docs to explore parameters and schemas interactively.

## Dashboard Usage

1. Open http://localhost:8501.
2. Navigate to **🚀 Launch Experiment** to generate comparison runs; the UI automatically schedules experiments for all agents.
3. Switch to **📊 Comparison Results**:
   - Filter by time range and minimum actions.
   - Quickly inspect the latest experiment per agent.
   - Review metric cards, bar charts, radar plots, conversion funnels, session statistics, action distributions, and reward timelines.
4. Use the sidebar to enable auto-refresh or export aggregated results as CSV.
