"""
Dashboard for comparing agents in E-commerce RL Recommendation System.
"""

import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def get_average_reward(results: Dict[str, Any]) -> float:
    """Return the most relevant average reward metric."""
    if results.get('current_average_reward') is not None:
        return results['current_average_reward']
    if results.get('cumulative_average_reward') is not None:
        return results['cumulative_average_reward']
    return results.get('average_reward', 0.0)

# Page configuration
st.set_page_config(
    page_title="🤖 RL Agents Comparison",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# State initialization
if 'comparison_experiments' not in st.session_state:
    st.session_state.comparison_experiments = []
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'last_comparison_id' not in st.session_state:
    st.session_state.last_comparison_id = None

# CSS Styles
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        animation: gradient-shift 3s ease infinite;
        margin-bottom: 2rem;
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    
    .agent-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .status-running { color: #00ff88; font-weight: bold; }
    .status-completed { color: #4CAF50; font-weight: bold; }
    .status-failed { color: #ff4444; font-weight: bold; }
    .status-pending { color: #FFA726; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def make_api_request(endpoint: str, method: str = "GET", data: Dict = None) -> tuple:
    """API request with error handling."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            return response.json(), True
        else:
            return {"error": f"HTTP {response.status_code}"}, False
    except Exception as e:
        return {"error": str(e)}, False

def launch_agent_comparison_experiment(config):
    """Launch comparison experiment for all agents."""
    agents = ["dqn", "epsilon_greedy", "linucb", "random"]
    agent_names = {
        "dqn": "Deep Q-Network",
        "epsilon_greedy": "Epsilon-Greedy",
        "linucb": "LinUCB",
        "random": "Random Baseline"
    }
    
    experiment_ids = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, agent in enumerate(agents):
        status_text.text(f"Launching experiment for {agent_names[agent]} agent...")
        
        experiment_config = {
            **config,
            "name": f"Agent Comparison - {agent_names[agent]}",
            "agent_type": agent
        }
        
        try:
            response, success = make_api_request("/experiments/start", "POST", experiment_config)
            if success and "experiment_id" in response:
                experiment_ids.append({
                    "id": response["experiment_id"],
                    "agent": agent,
                    "name": agent_names[agent],
                    "status": "running"
                })
                st.success(f"✅ {agent_names[agent]}: Started (ID: {response['experiment_id']})")
            else:
                st.error(f"❌ {agent_names[agent]}: Launch error - {response.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"❌ {agent_names[agent]}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(agents))
        time.sleep(0.5)
    
    status_text.text(f"Completed! Launched {len(experiment_ids)} experiments.")
    
    if experiment_ids:
        st.session_state.comparison_experiments = experiment_ids
        st.session_state.last_comparison_id = datetime.now().isoformat()
        st.info("🔄 Experiments started! Results will be updated automatically.")
        time.sleep(2)
        st.rerun()

def get_experiment_results():
    """Get results of all experiments."""
    experiments, success = make_api_request("/experiments/")
    
    if not success or not isinstance(experiments, list):
        return []
    
    # Filter only completed experiments with results
    completed_experiments = [
        e for e in experiments 
        if e.get('status') == 'completed' and e.get('results')
    ]
    
    return completed_experiments

def create_comprehensive_comparison_charts(experiments):
    """Create comprehensive agent comparison charts."""
    if not experiments:
        st.info("No completed experiments for comparison.")
        return
    
    # Prepare data
    comparison_data = []
    learning_curves = {}
    action_distributions = {}
    reward_distribution_rows = []
    conversion_records = []
    session_records = []
    reward_timelines = {}
    
    for exp in experiments:
        results = exp['results']
        config = exp['configuration']
        agent_type = config['agent_type']
        
        # Main metrics
        avg_reward = get_average_reward(results)
        cumulative_avg = results.get('cumulative_average_reward') or avg_reward
        current_avg = results.get('current_average_reward') or avg_reward
        performance_improvement = results.get('performance_improvement', 0.0)
        
        comparison_data.append({
            'Agent': agent_type.upper(),
            'Average Reward': avg_reward,
            'Cumulative Average': cumulative_avg,
            'Current Average': current_avg,
            'Improvement': performance_improvement,
            'Total Actions': results['total_actions'],
            'Users': results['total_users'],
            'Completion Time': results['completion_time'],
            'Actions/sec': results['total_actions'] / results['completion_time'] if results['completion_time'] > 0 else 0,
            'Efficiency': avg_reward * (results['total_actions'] / results['completion_time']) if results['completion_time'] > 0 else 0,
            'Products': config['n_products'],
            'Actions per User': config['actions_per_user']
        })
        
        # Learning curves
        if results.get('learning_curve'):
            curve = results['learning_curve']
            # Support both dict-based and numeric curves
            if curve and isinstance(curve[0], dict):
                learning_curves[agent_type] = [point.get('current_avg', point.get('cumulative_avg', 0)) for point in curve]
            else:
                # Legacy format - list of floats
                learning_curves[agent_type] = curve
        
        # Action distributions
        if results.get('action_distribution'):
            action_distributions[agent_type] = results['action_distribution']
        
        if results.get('reward_distribution'):
            for action_name, stats in results['reward_distribution'].items():
                reward_distribution_rows.append({
                    'Agent': agent_type.upper(),
                    'Action': action_name,
                    'Action Share': stats.get('percentage', 0) * 100,
                    'Average Reward': stats.get('avg_reward', 0),
                    'Count': stats.get('count', 0)
                })
        
        if results.get('conversion_metrics'):
            conversion_metrics = results['conversion_metrics']
            conversion_records.append({
                'Agent': agent_type.upper(),
                'Views': conversion_metrics.get('view_rate', 0),
                'Engagement': conversion_metrics.get('interaction_rate', 0),
                'Add to Cart': conversion_metrics.get('cart_rate', 0),
                'Purchases': conversion_metrics.get('purchase_rate', 0),
                'Negative Actions': conversion_metrics.get('negative_feedback_rate', 0)
            })
        
        if results.get('session_metrics'):
            session_metrics = results['session_metrics']
            session_records.append({
                'Agent': agent_type.upper(),
                'Sessions': session_metrics.get('sessions', 0),
                'Actions per Session': session_metrics.get('avg_actions_per_session', 0),
                'Reward per Session': session_metrics.get('avg_reward_per_session', 0.0),
                'Config Actions per User': session_metrics.get('configured_actions_per_user', 0),
                'Time per Session (s)': session_metrics.get('completion_time_per_session', 0.0)
            })
        
        # Reward timeline (supporting new structure)
        if results.get('reward_timeline'):
            timeline = results['reward_timeline']
            # Handle dict timeline entries
            if timeline and isinstance(timeline[0], dict):
                reward_timelines[agent_type] = timeline
            else:
                reward_timelines[agent_type] = timeline
    
    df = pd.DataFrame(comparison_data)
    
    # 1. Main metrics
    st.markdown("### 📊 Main Comparison Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_reward = df.loc[df['Average Reward'].idxmax()]
        st.markdown(f'<div class="metric-card">🏆 Best Reward<br><b>{best_reward["Agent"]}</b><br>{best_reward["Average Reward"]:.3f}</div>', unsafe_allow_html=True)
    
    with col2:
        fastest = df.loc[df['Actions/sec'].idxmax()]
        st.markdown(f'<div class="metric-card">⚡ Fastest<br><b>{fastest["Agent"]}</b><br>{fastest["Actions/sec"]:.1f} a/s</div>', unsafe_allow_html=True)
    
    with col3:
        most_efficient = df.loc[df['Efficiency'].idxmax()]
        st.markdown(f'<div class="metric-card">🎯 Most Efficient<br><b>{most_efficient["Agent"]}</b><br>{most_efficient["Efficiency"]:.2f}</div>', unsafe_allow_html=True)
    
    with col4:
        total_actions = df['Total Actions'].sum()
        st.markdown(f'<div class="metric-card">📈 Total Actions<br><b>{total_actions:,}</b><br>across all experiments</div>', unsafe_allow_html=True)
    
    # 2. Detailed comparison table
    st.markdown("### 📋 Detailed Comparison")
    st.dataframe(df)
    
    # 3. Comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Average reward
        fig_rewards = px.bar(
            df, x='Agent', y='Average Reward',
            color='Agent',
            title="🎯 Average Reward Comparison",
            text='Average Reward',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_rewards.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_rewards.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_rewards, config={'displayModeBar': False})
    
    with col2:
        # Performance
        fig_performance = px.bar(
            df, x='Agent', y='Actions/sec',
            color='Agent',
            title="⚡ Performance (actions/sec)",
            text='Actions/sec',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_performance.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_performance.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_performance, config={'displayModeBar': False})
    
    # 4. Efficiency vs Speed
    st.markdown("### 🎯 Efficiency vs Performance")
    fig_scatter = px.scatter(
        df, x='Actions/sec', y='Average Reward',
        size='Total Actions', color='Agent',
        title="Performance vs Quality Relationship",
        hover_data=['Completion Time', 'Users'],
        size_max=60
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, config={'displayModeBar': False})
    
    # 5. Learning curves
    if learning_curves:
        st.markdown("### 📈 Learning Curves")
        fig_learning = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for i, (agent, curve) in enumerate(learning_curves.items()):
            fig_learning.add_trace(go.Scatter(
                y=curve,
                mode='lines+markers',
                name=agent.upper(),
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=4)
            ))
        
        fig_learning.update_layout(
            title="Agent Learning Dynamics",
            xaxis_title="Training Step (x100 actions)",
            yaxis_title="Average Reward",
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig_learning, config={'displayModeBar': False})
    
    # 6. Radar chart
    st.markdown("### 🕸️ Multidimensional Comparison")
    
    # Normalize metrics for radar chart
    metrics = ['Average Reward', 'Actions/sec', 'Efficiency']
    fig_radar = go.Figure()
    
    colors = ['rgba(255, 107, 107, 0.6)', 'rgba(78, 205, 196, 0.6)', 
              'rgba(69, 183, 209, 0.6)', 'rgba(150, 206, 180, 0.6)']
    
    for i, agent in enumerate(df['Agent'].unique()):
        agent_data = df[df['Agent'] == agent].iloc[0]
        
        # Normalized values (0-1)
        values = []
        for metric in metrics:
            max_val = df[metric].max()
            min_val = df[metric].min()
            if max_val > min_val:
                normalized = (agent_data[metric] - min_val) / (max_val - min_val)
            else:
                normalized = 1.0
            values.append(normalized)
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the contour
            theta=metrics + [metrics[0]],
            fill='toself',
            name=agent,
            fillcolor=colors[i % len(colors)],
            line=dict(color=colors[i % len(colors)].replace('0.6', '1.0'), width=2)
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickmode='array',
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=['0%', '25%', '50%', '75%', '100%']
            )
        ),
        showlegend=True,
        title="Normalized Performance Comparison",
        height=600
    )
    st.plotly_chart(fig_radar, config={'displayModeBar': False})
    # 7. User action distribution
    if action_distributions:
        st.markdown("### 🎭 User Action Distribution")
        
        # Create subplots for each agent
        n_agents = len(action_distributions)
        cols = st.columns(min(n_agents, 2))
        
        for i, (agent, actions) in enumerate(action_distributions.items()):
            with cols[i % 2]:
                fig_pie = px.pie(
                    values=list(actions.values()),
                    names=list(actions.keys()),
                    title=f"Actions - {agent.upper()}"
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, config={'displayModeBar': False})
    
    # 8. Conversion metrics
    if conversion_records:
        st.markdown("### 🔁 Conversion Metrics")
        conv_df = pd.DataFrame(conversion_records)
        display_df = conv_df.copy()
        for col in display_df.columns:
            if col != 'Agent':
                display_df[col] = (display_df[col] * 100).round(2)
        st.dataframe(display_df.rename(columns=lambda c: c if c == 'Agent' else f"{c} (%)"))
        
        conv_melt = conv_df.melt(id_vars='Agent', var_name='Metric', value_name='Value')
        conv_melt['Value'] = conv_melt['Value'] * 100
        fig_conv = px.bar(
            conv_melt,
            x='Metric',
            y='Value',
            color='Agent',
            barmode='group',
            title="Conversion Rates at Each Step, %"
        )
        fig_conv.update_layout(height=450)
        st.plotly_chart(fig_conv, config={'displayModeBar': False})
    
    # 9. Session metrics
    if session_records:
        st.markdown("### 👥 User Session Metrics")
        session_df = pd.DataFrame(session_records)
        session_df['Actions per Session'] = session_df['Actions per Session'].round(2)
        session_df['Reward per Session'] = session_df['Reward per Session'].round(2)
        session_df['Time per Session (s)'] = session_df['Time per Session (s)'].round(2)
        st.dataframe(session_df)
        
        fig_sessions = px.bar(
            session_df,
            x='Agent',
            y='Actions per Session',
            color='Agent',
            title="Average Actions per Session",
            text='Actions per Session'
        )
        fig_sessions.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_sessions.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_sessions, config={'displayModeBar': False})
    
    # 10. Reward distribution by action
    if reward_distribution_rows:
        st.markdown("### 🎯 Rewards by Action")
        dist_df = pd.DataFrame(reward_distribution_rows)
        
        share_df = dist_df.copy()
        fig_dist = px.bar(
            share_df,
            x='Action Share',
            y='Agent',
            color='Action',
            orientation='h',
            barmode='stack',
            title="Action Type Distribution, %"
        )
        fig_dist.update_layout(height=500)
        st.plotly_chart(fig_dist, config={'displayModeBar': False})
        
        fig_reward = px.bar(
            dist_df,
            x='Action',
            y='Average Reward',
            color='Agent',
            barmode='group',
            title="Average Reward by Action"
        )
        fig_reward.update_layout(height=450)
        st.plotly_chart(fig_reward, config={'displayModeBar': False})
    
    # 11. Reward timeline
    if reward_timelines:
        st.markdown("### ⏱️ Reward Dynamics During Experiment")
        fig_timeline = go.Figure()
        colors = ['#FFB347', '#6A0572', '#2E86AB', '#4CAF50']
        for i, (agent, timeline) in enumerate(reward_timelines.items()):
            if not timeline:
                continue
            # Support new timeline schema containing current_avg_reward
            x_values = []
            y_values = []
            for point in timeline:
                if isinstance(point, dict):
                    x_values.append(point.get('actions', 0))
                    # Prefer current_avg_reward; fall back to avg_reward for legacy data
                    y_values.append(point.get('current_avg_reward') or point.get('avg_reward', 0))
                else:
                    # Legacy format
                    x_values.append(len(x_values) * 100)
                    y_values.append(point)
            
            fig_timeline.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines',
                name=agent.upper(),
                line=dict(color=colors[i % len(colors)], width=3)
            ))
        fig_timeline.update_layout(
            title="Average Reward Trend During Experiment",
            xaxis_title="Number of Actions",
            yaxis_title="Average Reward (rolling)",
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig_timeline, config={'displayModeBar': False})

def show_experiment_launcher():
    """Agent comparison experiment launch interface."""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("## 🚀 Launch Agent Comparison Experiment")
    
    with st.form("agent_comparison_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⚙️ Experiment Parameters")
            n_products = st.slider("Number of Products", 100, 2000, 500, step=50)
            n_users = st.slider("Number of Users", 10, 500, 100, step=10)
            actions_per_user = st.slider("Actions per User", 5, 50, 20, step=1)
            simulation_speed = st.slider("Simulation Speed", 0.5, 5.0, 2.0, step=0.1)
        
        with col2:
            st.markdown("### 🤖 Agents for Comparison")
            st.markdown("""
            **All 4 agents will be tested:**
            - 🧠 **Deep Q-Network (DQN)** - deep reinforcement learning
            - 🎯 **Epsilon-Greedy** - simple bandit algorithm
            - 📈 **LinUCB** - contextual bandit with linear model
            - 🎲 **Random Baseline** - random selection for comparison
            """)
            
            estimated_time = (n_users * actions_per_user * 4 * 0.1) / simulation_speed
            st.info(f"⏱️ Estimated execution time: ~{estimated_time:.1f} seconds")
        
        submitted = st.form_submit_button("🚀 Launch Agent Comparison")
        
        if submitted:
            config = {
                "description": "Automatic comparison of all agents",
                "n_products": n_products,
                "n_users": n_users,
                "actions_per_user": actions_per_user,
                "simulation_speed": simulation_speed
            }
            
            with st.spinner("Launching experiments for all agents..."):
                launch_agent_comparison_experiment(config)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_comparison_results():
    """Display agent comparison results."""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("## 📊 Agent Comparison Results")
    
    # Get all experiments
    experiments = get_experiment_results()
    
    if not experiments:
        st.info("No completed experiments. Launch agent comparison to get results.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Time filtering (recent experiments)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        time_filter = st.selectbox(
            "Experiment Period",
            ["Last Hour", "Last 6 Hours", "Last 24 Hours", "All Time"],
            index=1
        )
    
    with col2:
        min_actions = st.number_input("Minimum Actions", min_value=0, value=100, step=100)
    
    with col3:
        if st.button("🔄 Refresh Results"):
            st.rerun()
    
    # Apply filters
    now = datetime.now()
    time_deltas = {
        "Last Hour": timedelta(hours=1),
        "Last 6 Hours": timedelta(hours=6),
        "Last 24 Hours": timedelta(days=1),
        "All Time": timedelta(days=365)
    }
    
    cutoff_time = now - time_deltas[time_filter]
    
    filtered_experiments = []
    for exp in experiments:
        if exp.get('start_time'):
            try:
                start_time = datetime.fromisoformat(exp['start_time'].replace('Z', '+00:00'))
                if start_time.replace(tzinfo=None) >= cutoff_time:
                    if exp['results']['total_actions'] >= min_actions:
                        filtered_experiments.append(exp)
            except:
                continue
    
    if not filtered_experiments:
        st.warning(f"No experiments in selected period with minimum {min_actions} actions.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Group by agent types (take latest experiment of each type)
    latest_experiments = {}
    for exp in sorted(filtered_experiments, key=lambda x: x.get('start_time', ''), reverse=True):
        agent_type = exp['configuration']['agent_type']
        if agent_type not in latest_experiments:
            latest_experiments[agent_type] = exp
    
    final_experiments = list(latest_experiments.values())
    
    if len(final_experiments) < 2:
        st.warning("Not enough experiments for comparison. Need at least 2 different agents.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Display experiment status
    st.markdown("### 🎯 Latest Experiments Status")
    cols = st.columns(len(final_experiments))
    
    for i, exp in enumerate(final_experiments):
        with cols[i]:
            agent_name = exp['configuration']['agent_type'].upper()
            reward = get_average_reward(exp['results'])
            actions = exp['results']['total_actions']
            
            st.markdown(f"""
            <div class="agent-card">
                <h4>{agent_name}</h4>
                <p>Reward: {reward:.3f}</p>
                <p>Actions: {actions:,}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Create comprehensive comparison charts
    create_comprehensive_comparison_charts(final_experiments)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main dashboard application."""
    st.markdown('<h1 class="main-header">🤖 RL Agents Comparison</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Information
    st.sidebar.info("""
    **Dashboard Features:**
    - 🚀 Launch comparison of all agents
    - 📊 Detailed results analytics
    - 📈 Learning curves and metrics
    - 🔄 Automatic updates
    """)
    
    # Mode selection
    mode = st.sidebar.radio(
        "Select Mode:",
        ["🚀 Launch Experiment", "📊 Comparison Results"]
    )
    
    # Auto-refresh settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Settings")
    
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh
    
    if auto_refresh:
        refresh_interval = st.sidebar.slider("Interval (sec)", 10, 60, 30)
        st.sidebar.info(f"Refreshing every {refresh_interval} seconds")
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Export results
    st.sidebar.markdown("### 📤 Export")
    if st.sidebar.button("📊 Export Results"):
        experiments = get_experiment_results()
        if experiments:
            # Prepare data for export
            export_data = []
            for exp in experiments:
                results = exp['results']
                config = exp['configuration']
                export_data.append({
                    'agent_type': config['agent_type'],
                    'experiment_name': exp['name'],
                    'average_reward': get_average_reward(results),
                    'current_average_reward': results.get('current_average_reward'),
                    'total_actions': results['total_actions'],
                    'completion_time': results['completion_time'],
                    'n_users': config['n_users'],
                    'n_products': config['n_products'],
                    'start_time': exp.get('start_time', '')
                })
            
            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False)
            st.sidebar.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"agent_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.sidebar.warning("No data for export")
    
    # Main content
    if mode == "🚀 Launch Experiment":
        show_experiment_launcher()
    else:
        show_comparison_results()
    
    # Auto-refresh
    if auto_refresh and mode == "📊 Comparison Results":
        time.sleep(refresh_interval)
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"🤖 **RL Agents Comparison Dashboard** | "
        f"Last update: {datetime.now().strftime('%H:%M:%S')}"
    )

if __name__ == "__main__":
    main()
