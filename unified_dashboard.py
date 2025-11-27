"""
Дашборд сравнения агентов для E-commerce RL Recommendation System.
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

# Конфигурация
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Настройка страницы
st.set_page_config(
    page_title="🤖 Сравнение RL Агентов",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация состояния
if 'comparison_experiments' not in st.session_state:
    st.session_state.comparison_experiments = []
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'last_comparison_id' not in st.session_state:
    st.session_state.last_comparison_id = None

# Стили CSS
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
    """API запрос с обработкой ошибок."""
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
    """Запуск эксперимента сравнения всех агентов."""
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
        status_text.text(f"Запуск эксперимента для агента {agent_names[agent]}...")
        
        experiment_config = {
            **config,
            "name": f"Сравнение агентов - {agent_names[agent]}",
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
                st.success(f"✅ {agent_names[agent]}: Запущен (ID: {response['experiment_id']})")
            else:
                st.error(f"❌ {agent_names[agent]}: Ошибка запуска - {response.get('error', 'Неизвестная ошибка')}")
        
        except Exception as e:
            st.error(f"❌ {agent_names[agent]}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(agents))
        time.sleep(0.5)
    
    status_text.text(f"Завершено! Запущено {len(experiment_ids)} экспериментов.")
    
    if experiment_ids:
        st.session_state.comparison_experiments = experiment_ids
        st.session_state.last_comparison_id = datetime.now().isoformat()
        st.info("🔄 Эксперименты запущены! Результаты будут обновляться автоматически.")
        time.sleep(2)
        st.rerun()

def get_experiment_results():
    """Получение результатов всех экспериментов."""
    experiments, success = make_api_request("/experiments/")
    
    if not success or not isinstance(experiments, list):
        return []
    
    # Фильтруем только завершенные эксперименты с результатами
    completed_experiments = [
        e for e in experiments 
        if e.get('status') == 'completed' and e.get('results')
    ]
    
    return completed_experiments

def create_comprehensive_comparison_charts(experiments):
    """Создание комплексных графиков сравнения агентов."""
    if not experiments:
        st.info("Нет завершенных экспериментов для сравнения.")
        return
    
    # Подготовка данных
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
        
        # Основные метрики
        comparison_data.append({
            'Агент': agent_type.upper(),
            'Средняя награда': results['average_reward'],
            'Общие действия': results['total_actions'],
            'Пользователи': results['total_users'],
            'Время выполнения': results['completion_time'],
            'Действий/сек': results['total_actions'] / results['completion_time'] if results['completion_time'] > 0 else 0,
            'Эффективность': results['average_reward'] * (results['total_actions'] / results['completion_time']) if results['completion_time'] > 0 else 0,
            'Товары': config['n_products'],
            'Действий/пользователь': config['actions_per_user']
        })
        
        # Кривые обучения
        if results.get('learning_curve'):
            learning_curves[agent_type] = results['learning_curve']
        
        # Распределение действий
        if results.get('action_distribution'):
            action_distributions[agent_type] = results['action_distribution']
        
        if results.get('reward_distribution'):
            for action_name, stats in results['reward_distribution'].items():
                reward_distribution_rows.append({
                    'Агент': agent_type.upper(),
                    'Действие': action_name,
                    'Доля действий': stats.get('percentage', 0) * 100,
                    'Средняя награда': stats.get('avg_reward', 0),
                    'Количество': stats.get('count', 0)
                })
        
        if results.get('conversion_metrics'):
            conversion_metrics = results['conversion_metrics']
            conversion_records.append({
                'Агент': agent_type.upper(),
                'Просмотры': conversion_metrics.get('view_rate', 0),
                'Вовлеченность': conversion_metrics.get('interaction_rate', 0),
                'Добавление в корзину': conversion_metrics.get('cart_rate', 0),
                'Покупки': conversion_metrics.get('purchase_rate', 0),
                'Негативные действия': conversion_metrics.get('negative_feedback_rate', 0)
            })
        
        if results.get('session_metrics'):
            session_metrics = results['session_metrics']
            session_records.append({
                'Агент': agent_type.upper(),
                'Сессий': session_metrics.get('sessions', 0),
                'Действий/сессию': session_metrics.get('avg_actions_per_session', 0),
                'Награда/сессию': session_metrics.get('avg_reward_per_session', 0.0),
                'Конфиг. действий/польз.': session_metrics.get('configured_actions_per_user', 0),
                'Время/сессию (с)': session_metrics.get('completion_time_per_session', 0.0)
            })
        
        if results.get('reward_timeline'):
            reward_timelines[agent_type] = results['reward_timeline']
    
    df = pd.DataFrame(comparison_data)
    
    # 1. Основные метрики
    st.markdown("### 📊 Основные метрики сравнения")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_reward = df.loc[df['Средняя награда'].idxmax()]
        st.markdown(f'<div class="metric-card">🏆 Лучшая награда<br><b>{best_reward["Агент"]}</b><br>{best_reward["Средняя награда"]:.3f}</div>', unsafe_allow_html=True)
    
    with col2:
        fastest = df.loc[df['Действий/сек'].idxmax()]
        st.markdown(f'<div class="metric-card">⚡ Самый быстрый<br><b>{fastest["Агент"]}</b><br>{fastest["Действий/сек"]:.1f} д/с</div>', unsafe_allow_html=True)
    
    with col3:
        most_efficient = df.loc[df['Эффективность'].idxmax()]
        st.markdown(f'<div class="metric-card">🎯 Самый эффективный<br><b>{most_efficient["Агент"]}</b><br>{most_efficient["Эффективность"]:.2f}</div>', unsafe_allow_html=True)
    
    with col4:
        total_actions = df['Общие действия'].sum()
        st.markdown(f'<div class="metric-card">📈 Всего действий<br><b>{total_actions:,}</b><br>во всех экспериментах</div>', unsafe_allow_html=True)
    
    # 2. Детальная таблица сравнения
    st.markdown("### 📋 Детальное сравнение")
    st.dataframe(df)
    
    # 3. Графики сравнения
    col1, col2 = st.columns(2)
    
    with col1:
        # Средняя награда
        fig_rewards = px.bar(
            df, x='Агент', y='Средняя награда',
            color='Агент',
            title="🎯 Сравнение средних наград",
            text='Средняя награда',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_rewards.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_rewards.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_rewards, config={'displayModeBar': False})
    
    with col2:
        # Производительность
        fig_performance = px.bar(
            df, x='Агент', y='Действий/сек',
            color='Агент',
            title="⚡ Производительность (действий/сек)",
            text='Действий/сек',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_performance.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_performance.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_performance, config={'displayModeBar': False})
    
    # 4. Эффективность vs Скорость
    st.markdown("### 🎯 Эффективность vs Производительность")
    fig_scatter = px.scatter(
        df, x='Действий/сек', y='Средняя награда',
        size='Общие действия', color='Агент',
        title="Соотношение производительности и качества",
        hover_data=['Время выполнения', 'Пользователи'],
        size_max=60
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, config={'displayModeBar': False})
    
    # 5. Кривые обучения
    if learning_curves:
        st.markdown("### 📈 Кривые обучения")
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
            title="Динамика обучения агентов",
            xaxis_title="Шаг обучения (x100 действий)",
            yaxis_title="Средняя награда",
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig_learning, config={'displayModeBar': False})
    
    # 6. Радарная диаграмма
    st.markdown("### 🕸️ Многомерное сравнение")
    
    # Нормализация метрик для радарной диаграммы
    metrics = ['Средняя награда', 'Действий/сек', 'Эффективность']
    fig_radar = go.Figure()
    
    colors = ['rgba(255, 107, 107, 0.6)', 'rgba(78, 205, 196, 0.6)', 
              'rgba(69, 183, 209, 0.6)', 'rgba(150, 206, 180, 0.6)']
    
    for i, agent in enumerate(df['Агент'].unique()):
        agent_data = df[df['Агент'] == agent].iloc[0]
        
        # Нормализованные значения (0-1)
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
            r=values + [values[0]],  # Замыкаем контур
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
        title="Нормализованное сравнение производительности",
        height=600
    )
    st.plotly_chart(fig_radar, config={'displayModeBar': False})
    
    # 7. Распределение действий пользователей
    if action_distributions:
        st.markdown("### 🎭 Распределение действий пользователей")
        
        # Создаем подграфики для каждого агента
        n_agents = len(action_distributions)
        cols = st.columns(min(n_agents, 2))
        
        for i, (agent, actions) in enumerate(action_distributions.items()):
            with cols[i % 2]:
                fig_pie = px.pie(
                    values=list(actions.values()),
                    names=list(actions.keys()),
                    title=f"Действия - {agent.upper()}"
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, config={'displayModeBar': False})
    
    # 8. Конверсионные метрики
    if conversion_records:
        st.markdown("### 🔁 Конверсионные метрики")
        conv_df = pd.DataFrame(conversion_records)
        display_df = conv_df.copy()
        for col in display_df.columns:
            if col != 'Агент':
                display_df[col] = (display_df[col] * 100).round(2)
        st.dataframe(display_df.rename(columns=lambda c: c if c == 'Агент' else f"{c} (%)"))
        
        conv_melt = conv_df.melt(id_vars='Агент', var_name='Метрика', value_name='Значение')
        conv_melt['Значение'] = conv_melt['Значение'] * 100
        fig_conv = px.bar(
            conv_melt,
            x='Метрика',
            y='Значение',
            color='Агент',
            barmode='group',
            title="Конверсии на каждом шаге, %"
        )
        fig_conv.update_layout(height=450)
        st.plotly_chart(fig_conv, config={'displayModeBar': False})
    
    # 9. Метрики сессий
    if session_records:
        st.markdown("### 👥 Метрики пользовательских сессий")
        session_df = pd.DataFrame(session_records)
        session_df['Действий/сессию'] = session_df['Действий/сессию'].round(2)
        session_df['Награда/сессию'] = session_df['Награда/сессию'].round(2)
        session_df['Время/сессию (с)'] = session_df['Время/сессию (с)'].round(2)
        st.dataframe(session_df)
        
        fig_sessions = px.bar(
            session_df,
            x='Агент',
            y='Действий/сессию',
            color='Агент',
            title="Среднее количество действий в сессии",
            text='Действий/сессию'
        )
        fig_sessions.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_sessions.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_sessions, config={'displayModeBar': False})
    
    # 10. Распределение наград по действиям
    if reward_distribution_rows:
        st.markdown("### 🎯 Награды по действиям")
        dist_df = pd.DataFrame(reward_distribution_rows)
        
        share_df = dist_df.copy()
        fig_dist = px.bar(
            share_df,
            x='Доля действий',
            y='Агент',
            color='Действие',
            orientation='h',
            barmode='stack',
            title="Доля действий по типам, %"
        )
        fig_dist.update_layout(height=500)
        st.plotly_chart(fig_dist, config={'displayModeBar': False})
        
        fig_reward = px.bar(
            dist_df,
            x='Действие',
            y='Средняя награда',
            color='Агент',
            barmode='group',
            title="Средняя награда по действиям"
        )
        fig_reward.update_layout(height=450)
        st.plotly_chart(fig_reward, config={'displayModeBar': False})
    
    # 11. Таймлайн наград
    if reward_timelines:
        st.markdown("### ⏱️ Динамика награды в ходе эксперимента")
        fig_timeline = go.Figure()
        colors = ['#FFB347', '#6A0572', '#2E86AB', '#4CAF50']
        for i, (agent, timeline) in enumerate(reward_timelines.items()):
            if not timeline:
                continue
            fig_timeline.add_trace(go.Scatter(
                x=[point.get('actions', 0) for point in timeline],
                y=[point.get('avg_reward', 0) for point in timeline],
                mode='lines',
                name=agent.upper(),
                line=dict(color=colors[i % len(colors)], width=3)
            ))
        fig_timeline.update_layout(
            title="Изменение средней награды по мере выполнения действий",
            xaxis_title="Количество действий",
            yaxis_title="Средняя награда",
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig_timeline, config={'displayModeBar': False})

def show_experiment_launcher():
    """Интерфейс запуска эксперимента сравнения агентов."""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("## 🚀 Запуск эксперимента сравнения агентов")
    
    with st.form("agent_comparison_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⚙️ Параметры эксперимента")
            n_products = st.slider("Количество товаров", 100, 2000, 500, step=50)
            n_users = st.slider("Количество пользователей", 10, 500, 100, step=10)
            actions_per_user = st.slider("Действий на пользователя", 5, 50, 20, step=1)
            simulation_speed = st.slider("Скорость симуляции", 0.5, 5.0, 2.0, step=0.1)
        
        with col2:
            st.markdown("### 🤖 Агенты для сравнения")
            st.markdown("""
            **Будут протестированы все 4 агента:**
            - 🧠 **Deep Q-Network (DQN)** - глубокое обучение с подкреплением
            - 🎯 **Epsilon-Greedy** - простой bandit алгоритм
            - 📈 **LinUCB** - контекстуальный bandit с линейной моделью
            - 🎲 **Random Baseline** - случайный выбор для сравнения
            """)
            
            estimated_time = (n_users * actions_per_user * 4 * 0.1) / simulation_speed
            st.info(f"⏱️ Ожидаемое время выполнения: ~{estimated_time:.1f} секунд")
        
        submitted = st.form_submit_button("🚀 Запустить сравнение агентов")
        
        if submitted:
            config = {
                "description": "Автоматическое сравнение всех агентов",
                "n_products": n_products,
                "n_users": n_users,
                "actions_per_user": actions_per_user,
                "simulation_speed": simulation_speed
            }
            
            with st.spinner("Запуск экспериментов для всех агентов..."):
                launch_agent_comparison_experiment(config)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_comparison_results():
    """Отображение результатов сравнения агентов."""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("## 📊 Результаты сравнения агентов")
    
    # Получение всех экспериментов
    experiments = get_experiment_results()
    
    if not experiments:
        st.info("Нет завершенных экспериментов. Запустите сравнение агентов для получения результатов.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Фильтрация по времени (последние эксперименты)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        time_filter = st.selectbox(
            "Период экспериментов",
            ["Последний час", "Последние 6 часов", "Последние 24 часа", "Все время"],
            index=1
        )
    
    with col2:
        min_actions = st.number_input("Минимум действий", min_value=0, value=100, step=100)
    
    with col3:
        if st.button("🔄 Обновить результаты"):
            st.rerun()
    
    # Применение фильтров
    now = datetime.now()
    time_deltas = {
        "Последний час": timedelta(hours=1),
        "Последние 6 часов": timedelta(hours=6),
        "Последние 24 часа": timedelta(days=1),
        "Все время": timedelta(days=365)
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
        st.warning(f"Нет экспериментов за выбранный период с минимумом {min_actions} действий.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Группировка по типам агентов (берем последний эксперимент каждого типа)
    latest_experiments = {}
    for exp in sorted(filtered_experiments, key=lambda x: x.get('start_time', ''), reverse=True):
        agent_type = exp['configuration']['agent_type']
        if agent_type not in latest_experiments:
            latest_experiments[agent_type] = exp
    
    final_experiments = list(latest_experiments.values())
    
    if len(final_experiments) < 2:
        st.warning("Недостаточно экспериментов для сравнения. Нужно минимум 2 разных агента.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Отображение статуса экспериментов
    st.markdown("### 🎯 Статус последних экспериментов")
    cols = st.columns(len(final_experiments))
    
    for i, exp in enumerate(final_experiments):
        with cols[i]:
            agent_name = exp['configuration']['agent_type'].upper()
            reward = exp['results']['average_reward']
            actions = exp['results']['total_actions']
            
            st.markdown(f"""
            <div class="agent-card">
                <h4>{agent_name}</h4>
                <p>Награда: {reward:.3f}</p>
                <p>Действий: {actions:,}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Создание комплексных графиков сравнения
    create_comprehensive_comparison_charts(final_experiments)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Главное приложение дашборда."""
    st.markdown('<h1 class="main-header">🤖 Сравнение RL Агентов</h1>', unsafe_allow_html=True)
    
    # Боковая панель
    st.sidebar.title("🎛️ Управление")
    
    # Информация
    st.sidebar.info("""
    **Функции дашборда:**
    - 🚀 Запуск сравнения всех агентов
    - 📊 Детальная аналитика результатов
    - 📈 Кривые обучения и метрики
    - 🔄 Автоматическое обновление
    """)
    
    # Выбор режима
    mode = st.sidebar.radio(
        "Выберите режим:",
        ["🚀 Запуск эксперимента", "📊 Результаты сравнения"]
    )
    
    # Настройки автообновления
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Настройки")
    
    auto_refresh = st.sidebar.checkbox("🔄 Автообновление", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh
    
    if auto_refresh:
        refresh_interval = st.sidebar.slider("Интервал (сек)", 10, 60, 30)
        st.sidebar.info(f"Обновление каждые {refresh_interval} секунд")
    
    # Кнопка ручного обновления
    if st.sidebar.button("🔄 Обновить сейчас"):
        st.rerun()
    
    # Экспорт результатов
    st.sidebar.markdown("### 📤 Экспорт")
    if st.sidebar.button("📊 Экспорт результатов"):
        experiments = get_experiment_results()
        if experiments:
            # Подготовка данных для экспорта
            export_data = []
            for exp in experiments:
                results = exp['results']
                config = exp['configuration']
                export_data.append({
                    'agent_type': config['agent_type'],
                    'experiment_name': exp['name'],
                    'average_reward': results['average_reward'],
                    'total_actions': results['total_actions'],
                    'completion_time': results['completion_time'],
                    'n_users': config['n_users'],
                    'n_products': config['n_products'],
                    'start_time': exp.get('start_time', '')
                })
            
            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False)
            st.sidebar.download_button(
                label="Скачать CSV",
                data=csv,
                file_name=f"agent_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.sidebar.warning("Нет данных для экспорта")
    
    # Основной контент
    if mode == "🚀 Запуск эксперимента":
        show_experiment_launcher()
    else:
        show_comparison_results()
    
    # Автообновление
    if auto_refresh and mode == "📊 Результаты сравнения":
        time.sleep(refresh_interval)
        st.rerun()
    
    # Подвал
    st.markdown("---")
    st.markdown(
        f"🤖 **Дашборд сравнения RL агентов** | "
        f"Последнее обновление: {datetime.now().strftime('%H:%M:%S')}"
    )

if __name__ == "__main__":
    main()
