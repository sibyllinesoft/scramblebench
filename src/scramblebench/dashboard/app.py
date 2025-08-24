"""
ScrambleBench Dashboard - Interactive Analysis of Evaluation Results

A Streamlit application for exploring and visualizing ScrambleBench evaluation runs.
Takes a run_id parameter and provides comprehensive analysis dashboards.

Usage:
    streamlit run src/scramblebench/dashboard/app.py -- --run-id {{RUN_ID}}
"""

import os
import sys
import argparse
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Add the src directory to the Python path to enable imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from scramblebench.db.session import get_database_manager, DatabaseManager
from scramblebench.db.repository import RepositoryFactory
from scramblebench.db.models import Run, Evaluation, Aggregate
from scramblebench.dashboard.config import (
    DASHBOARD_CONFIG, CACHE_SETTINGS, CHART_COLORS, NAVIGATION_SECTIONS,
    DEFAULT_LIMITS, format_metric, get_status_emoji, get_boolean_emoji
)

logger = logging.getLogger(__name__)

# Configure Streamlit page settings
st.set_page_config(**DASHBOARD_CONFIG)


@st.cache_resource
def get_db_manager() -> DatabaseManager:
    """Get database manager with caching."""
    return get_database_manager()


@st.cache_resource
def get_repositories(_db_manager: DatabaseManager) -> RepositoryFactory:
    """Get repository factory with caching."""
    return RepositoryFactory(_db_manager)


@st.cache_data(ttl=CACHE_SETTINGS['data_ttl'])
def load_run_data(run_id: str) -> Optional[Dict[str, Any]]:
    """Load run data with caching."""
    try:
        db_manager = get_db_manager()
        repos = get_repositories(db_manager)
        
        run_repo = repos.get_run_repository()
        run = run_repo.get_by_run_id(run_id)
        
        if not run:
            return None
            
        return {
            'run_id': run.run_id,
            'started_at': run.started_at,
            'completed_at': run.completed_at,
            'status': run.status,
            'total_evaluations': run.total_evaluations,
            'completed_evaluations': run.completed_evaluations,
            'config_yaml': run.config_yaml,
            'config_hash': run.config_hash,
            'seed': run.seed,
            'git_sha': run.git_sha,
            'progress_percentage': run.progress_percentage,
            'duration': run.duration,
            'is_completed': run.is_completed
        }
    except Exception as e:
        logger.error(f"Error loading run data: {e}")
        return None


@st.cache_data(ttl=CACHE_SETTINGS['data_ttl'])
def load_evaluation_stats(run_id: str) -> Dict[str, Any]:
    """Load evaluation statistics with caching."""
    try:
        db_manager = get_db_manager()
        repos = get_repositories(db_manager)
        
        eval_repo = repos.get_evaluation_repository()
        return eval_repo.get_evaluation_stats(run_id)
    except Exception as e:
        logger.error(f"Error loading evaluation stats: {e}")
        return {}


@st.cache_data(ttl=CACHE_SETTINGS['data_ttl'])
def load_model_performance(run_id: str) -> List[Dict[str, Any]]:
    """Load model performance data with caching."""
    try:
        db_manager = get_db_manager()
        repos = get_repositories(db_manager)
        
        eval_repo = repos.get_evaluation_repository()
        return eval_repo.get_model_performance(run_id)
    except Exception as e:
        logger.error(f"Error loading model performance: {e}")
        return []


@st.cache_data(ttl=CACHE_SETTINGS['data_ttl'])
def load_aggregate_data(run_id: str) -> List[Dict[str, Any]]:
    """Load aggregate analysis data with caching."""
    try:
        db_manager = get_db_manager()
        repos = get_repositories(db_manager)
        
        agg_repo = repos.get_aggregate_repository()
        return agg_repo.get_model_comparison(run_id)
    except Exception as e:
        logger.error(f"Error loading aggregate data: {e}")
        return []


@st.cache_data(ttl=CACHE_SETTINGS['data_ttl'])
def load_evaluations_sample(run_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Load sample of evaluations for detailed inspection."""
    try:
        db_manager = get_db_manager()
        repos = get_repositories(db_manager)
        
        eval_repo = repos.get_evaluation_repository()
        evaluations = eval_repo.get_by_run(run_id, limit=limit or DEFAULT_LIMITS['evaluation_sample'])
        
        return [
            {
                'eval_id': eval.eval_id,
                'item_id': eval.item_id,
                'model_id': eval.model_id,
                'transform': eval.transform,
                'is_correct': eval.is_correct,
                'latency_ms': eval.latency_ms,
                'cost_usd': eval.cost_usd,
                'timestamp': eval.timestamp
            }
            for eval in evaluations
        ]
    except Exception as e:
        logger.error(f"Error loading evaluations sample: {e}")
        return []


@st.cache_data(ttl=CACHE_SETTINGS['long_data_ttl'])
def load_available_runs() -> List[Dict[str, Any]]:
    """Load list of available runs."""
    try:
        db_manager = get_db_manager()
        repos = get_repositories(db_manager)
        
        run_repo = repos.get_run_repository()
        runs = run_repo.get_completed_runs(limit=DEFAULT_LIMITS['recent_runs'])
        
        return [
            {
                'run_id': run.run_id,
                'started_at': run.started_at,
                'completed_at': run.completed_at,
                'status': run.status,
                'progress_percentage': run.progress_percentage
            }
            for run in runs
        ]
    except Exception as e:
        logger.error(f"Error loading available runs: {e}")
        return []


def render_run_overview(run_data: Dict[str, Any], eval_stats: Dict[str, Any]):
    """Render the run overview section."""
    st.header("📊 Run Overview")
    
    # Run metadata in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Status", run_data.get('status', 'Unknown'))
        st.metric("Seed", run_data.get('seed', 'N/A'))
    
    with col2:
        progress = run_data.get('progress_percentage', 0)
        st.metric("Progress", f"{progress:.1f}%")
        
        if run_data.get('duration'):
            st.metric("Duration", f"{run_data['duration']:.0f}s")
        else:
            st.metric("Duration", "Running...")
    
    with col3:
        total_evals = eval_stats.get('total_evaluations', 0)
        completed_evals = eval_stats.get('completed_evaluations', 0)
        st.metric("Total Evaluations", total_evals)
        st.metric("Completed", completed_evals)
    
    with col4:
        accuracy = eval_stats.get('accuracy', 0) * 100
        st.metric("Overall Accuracy", f"{accuracy:.1f}%")
        
        avg_latency = eval_stats.get('avg_latency_ms', 0)
        st.metric("Avg Latency", f"{avg_latency:.0f}ms")
    
    # Timeline
    st.subheader("⏰ Timeline")
    started_at = run_data.get('started_at')
    completed_at = run_data.get('completed_at')
    
    if started_at:
        st.write(f"**Started:** {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if completed_at:
        st.write(f"**Completed:** {completed_at.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration preview
    with st.expander("🔧 Configuration Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Run ID:** {run_data.get('run_id', 'N/A')}")
            st.write(f"**Config Hash:** {run_data.get('config_hash', 'N/A')}")
            st.write(f"**Git SHA:** {run_data.get('git_sha', 'N/A')}")
        
        with col2:
            config_yaml = run_data.get('config_yaml', '')
            if config_yaml:
                st.code(config_yaml[:500] + ('...' if len(config_yaml) > 500 else ''), 
                       language='yaml')


def render_model_comparison(model_performance: List[Dict[str, Any]]):
    """Render model comparison charts."""
    st.header("🤖 Model Performance Comparison")
    
    if not model_performance:
        st.warning("No model performance data available.")
        return
    
    df = pd.DataFrame(model_performance)
    
    # Accuracy comparison
    st.subheader("Accuracy by Model")
    
    fig_acc = px.bar(
        df, 
        x='model_id', 
        y='accuracy',
        title='Model Accuracy Comparison',
        labels={'accuracy': 'Accuracy (%)', 'model_id': 'Model ID'},
        color='accuracy',
        color_continuous_scale=CHART_COLORS['accuracy']
    )
    fig_acc.update_layout(showlegend=False)
    fig_acc.update_traces(texttemplate='%{y:.1%}', textposition='outside')
    st.plotly_chart(fig_acc, use_container_width=True)
    
    # Performance metrics comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Latency")
        fig_lat = px.bar(
            df, 
            x='model_id', 
            y='avg_latency_ms',
            title='Average Latency by Model',
            labels={'avg_latency_ms': 'Latency (ms)', 'model_id': 'Model ID'},
            color='avg_latency_ms',
            color_continuous_scale=CHART_COLORS['latency']
        )
        fig_lat.update_layout(showlegend=False)
        st.plotly_chart(fig_lat, use_container_width=True)
    
    with col2:
        st.subheader("Total Cost")
        fig_cost = px.bar(
            df, 
            x='model_id', 
            y='total_cost_usd',
            title='Total Cost by Model',
            labels={'total_cost_usd': 'Cost (USD)', 'model_id': 'Model ID'},
            color='total_cost_usd',
            color_continuous_scale=CHART_COLORS['cost']
        )
        fig_cost.update_layout(showlegend=False)
        st.plotly_chart(fig_cost, use_container_width=True)
    
    # Performance table
    st.subheader("📋 Detailed Performance Metrics")
    df_display = df.copy()
    df_display['accuracy'] = (df_display['accuracy'] * 100).round(2).astype(str) + '%'
    df_display['avg_latency_ms'] = df_display['avg_latency_ms'].round(0).astype(int)
    df_display['total_cost_usd'] = '$' + df_display['total_cost_usd'].round(4).astype(str)
    
    st.dataframe(
        df_display[['model_id', 'total_evaluations', 'correct_evaluations', 
                   'accuracy', 'avg_latency_ms', 'total_cost_usd']],
        use_container_width=True
    )


def render_transform_analysis(aggregate_data: List[Dict[str, Any]]):
    """Render transform robustness analysis."""
    st.header("🔄 Transform Analysis")
    
    if not aggregate_data:
        st.warning("No aggregate analysis data available.")
        return
    
    df = pd.DataFrame(aggregate_data)
    
    # Transform comparison heatmap
    st.subheader("Robustness Heatmap")
    
    # Pivot data for heatmap
    heatmap_data = df.pivot_table(
        index='model_id', 
        columns='transform', 
        values='avg_accuracy',
        fill_value=0
    )
    
    fig_heatmap = px.imshow(
        heatmap_data,
        title='Model Performance Across Transforms',
        labels=dict(x="Transform", y="Model", color="Accuracy"),
        color_continuous_scale=CHART_COLORS['robustness'],
        aspect='auto'
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Transform robustness metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average RRS (Relative Robustness Score)")
        if 'avg_rrs' in df.columns and df['avg_rrs'].notna().any():
            rrs_data = df.dropna(subset=['avg_rrs'])
            fig_rrs = px.box(
                rrs_data, 
                x='transform', 
                y='avg_rrs',
                title='RRS Distribution by Transform'
            )
            st.plotly_chart(fig_rrs, use_container_width=True)
        else:
            st.info("RRS data not available for this run.")
    
    with col2:
        st.subheader("Language Dependency Coefficient (LDC)")
        if 'avg_ldc' in df.columns and df['avg_ldc'].notna().any():
            ldc_data = df.dropna(subset=['avg_ldc'])
            fig_ldc = px.scatter(
                ldc_data, 
                x='avg_accuracy', 
                y='avg_ldc',
                color='model_id',
                title='Accuracy vs Language Dependency'
            )
            st.plotly_chart(fig_ldc, use_container_width=True)
        else:
            st.info("LDC data not available for this run.")
    
    # Best performing transforms
    st.subheader("🏆 Best Performing Configurations")
    
    # Sort by accuracy and show top performers
    top_configs = df.nlargest(DEFAULT_LIMITS['top_performers'], 'avg_accuracy')
    
    fig_top = px.bar(
        top_configs,
        x='avg_accuracy',
        y=[f"{row['model_id']}_{row['transform']}" for _, row in top_configs.iterrows()],
        orientation='h',
        title='Top 10 Model-Transform Combinations',
        labels={'avg_accuracy': 'Average Accuracy', 'y': 'Model + Transform'}
    )
    st.plotly_chart(fig_top, use_container_width=True)


def render_evaluation_details(evaluations_sample: List[Dict[str, Any]]):
    """Render detailed evaluation inspection."""
    st.header("🔍 Evaluation Details")
    
    if not evaluations_sample:
        st.warning("No evaluation data available.")
        return
    
    df = pd.DataFrame(evaluations_sample)
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        correct_rate = (df['is_correct'].sum() / len(df)) * 100 if len(df) > 0 else 0
        st.metric("Sample Accuracy", f"{correct_rate:.1f}%")
    
    with col2:
        avg_latency = df['latency_ms'].mean() if len(df) > 0 else 0
        st.metric("Avg Latency", f"{avg_latency:.0f}ms")
    
    with col3:
        total_cost = df['cost_usd'].sum() if len(df) > 0 else 0
        st.metric("Sample Cost", f"${total_cost:.4f}")
    
    with col4:
        unique_models = df['model_id'].nunique() if len(df) > 0 else 0
        st.metric("Models Tested", unique_models)
    
    # Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Latency Distribution")
        if len(df) > 0:
            fig_lat_dist = px.histogram(
                df, 
                x='latency_ms',
                nbins=30,
                title='Latency Distribution',
                labels={'latency_ms': 'Latency (ms)', 'count': 'Count'}
            )
            st.plotly_chart(fig_lat_dist, use_container_width=True)
    
    with col2:
        st.subheader("Cost per Evaluation")
        if len(df) > 0:
            fig_cost_dist = px.histogram(
                df, 
                x='cost_usd',
                nbins=30,
                title='Cost Distribution',
                labels={'cost_usd': 'Cost (USD)', 'count': 'Count'}
            )
            st.plotly_chart(fig_cost_dist, use_container_width=True)
    
    # Accuracy by transform
    if len(df) > 0:
        st.subheader("Accuracy by Transform")
        accuracy_by_transform = df.groupby('transform')['is_correct'].agg(['mean', 'count']).reset_index()
        accuracy_by_transform['accuracy_pct'] = accuracy_by_transform['mean'] * 100
        
        fig_transform_acc = px.bar(
            accuracy_by_transform,
            x='transform',
            y='accuracy_pct',
            title='Accuracy by Transform Type',
            labels={'accuracy_pct': 'Accuracy (%)', 'transform': 'Transform'},
            color='accuracy_pct',
            color_continuous_scale=CHART_COLORS['accuracy']
        )
        fig_transform_acc.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
        st.plotly_chart(fig_transform_acc, use_container_width=True)
    
    # Data table with filtering
    st.subheader("📋 Sample Evaluations")
    
    if len(df) > 0:
        # Add filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_filter = st.selectbox(
                "Filter by Model",
                options=['All'] + sorted(df['model_id'].unique().tolist())
            )
        
        with col2:
            transform_filter = st.selectbox(
                "Filter by Transform",
                options=['All'] + sorted(df['transform'].unique().tolist())
            )
        
        with col3:
            correct_filter = st.selectbox(
                "Filter by Result",
                options=['All', 'Correct Only', 'Incorrect Only']
            )
        
        # Apply filters
        filtered_df = df.copy()
        if model_filter != 'All':
            filtered_df = filtered_df[filtered_df['model_id'] == model_filter]
        if transform_filter != 'All':
            filtered_df = filtered_df[filtered_df['transform'] == transform_filter]
        if correct_filter == 'Correct Only':
            filtered_df = filtered_df[filtered_df['is_correct'] == True]
        elif correct_filter == 'Incorrect Only':
            filtered_df = filtered_df[filtered_df['is_correct'] == False]
        
        # Format display data
        display_df = filtered_df.copy()
        display_df['is_correct'] = display_df['is_correct'].map(get_boolean_emoji)
        display_df['latency_ms'] = display_df['latency_ms'].round(0).astype(int)
        display_df['cost_usd'] = display_df['cost_usd'].round(6)
        
        st.dataframe(
            display_df[['eval_id', 'item_id', 'model_id', 'transform', 
                       'is_correct', 'latency_ms', 'cost_usd', 'timestamp']],
            use_container_width=True
        )


def render_sidebar(available_runs: List[Dict[str, Any]], current_run_id: Optional[str]):
    """Render sidebar with run selection and navigation."""
    st.sidebar.title("🔬 ScrambleBench Dashboard")
    
    # Run selection
    st.sidebar.header("📊 Run Selection")
    
    if available_runs:
        run_options = [f"{run['run_id']} ({run['status']})" for run in available_runs]
        run_ids = [run['run_id'] for run in available_runs]
        
        # Find current selection index
        current_idx = 0
        if current_run_id and current_run_id in run_ids:
            current_idx = run_ids.index(current_run_id)
        
        selected_idx = st.sidebar.selectbox(
            "Select Run",
            range(len(run_options)),
            format_func=lambda x: run_options[x],
            index=current_idx
        )
        
        selected_run_id = run_ids[selected_idx]
        
        if selected_run_id != current_run_id:
            st.experimental_set_query_params(run_id=selected_run_id)
            st.experimental_rerun()
    else:
        st.sidebar.warning("No completed runs found.")
    
    # Navigation
    st.sidebar.header("🧭 Navigation")
    selected_section = st.sidebar.radio("Jump to Section", NAVIGATION_SECTIONS)
    
    # Database health check
    st.sidebar.header("🔧 System Status")
    
    try:
        db_manager = get_db_manager()
        health = db_manager.health_check()
        
        if health.get('status') == 'healthy':
            st.sidebar.success("✅ Database Connected")
        else:
            st.sidebar.error("❌ Database Issues")
            st.sidebar.write(health.get('error', 'Unknown error'))
    except Exception as e:
        st.sidebar.error(f"❌ Database Error: {e}")
    
    # Refresh data button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    return selected_section


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ScrambleBench Dashboard")
    parser.add_argument('--run-id', type=str, help='Run ID to display')
    
    # Parse known args to handle Streamlit's additional arguments
    args, unknown = parser.parse_known_args()
    return args


def main():
    """Main dashboard application."""
    # Parse arguments
    args = parse_args()
    
    # Get run_id from command line args or query params
    run_id = None
    if args.run_id:
        run_id = args.run_id
    else:
        # Try to get from query params
        query_params = st.experimental_get_query_params()
        if 'run_id' in query_params:
            run_id = query_params['run_id'][0]
    
    # Load available runs for sidebar
    available_runs = load_available_runs()
    
    # If no run_id specified and runs are available, use the first one
    if not run_id and available_runs:
        run_id = available_runs[0]['run_id']
        st.experimental_set_query_params(run_id=run_id)
    
    # Render sidebar
    selected_section = render_sidebar(available_runs, run_id)
    
    # Main content area
    if not run_id:
        st.title("🔬 ScrambleBench Dashboard")
        st.error("No run ID specified. Please provide a run_id via command line or select from the sidebar.")
        st.markdown("""
        **Usage:**
        ```bash
        streamlit run src/scramblebench/dashboard/app.py -- --run-id {{RUN_ID}}
        ```
        """)
        return
    
    # Load data for the selected run
    with st.spinner(f"Loading data for run {run_id}..."):
        run_data = load_run_data(run_id)
    
    if not run_data:
        st.error(f"Run '{run_id}' not found in the database.")
        return
    
    # Display title with run info
    st.title(f"🔬 ScrambleBench Dashboard - {run_id}")
    st.markdown(f"**Status:** {run_data.get('status', 'Unknown')} | "
                f"**Progress:** {run_data.get('progress_percentage', 0):.1f}%")
    
    # Load additional data
    eval_stats = load_evaluation_stats(run_id)
    model_performance = load_model_performance(run_id)
    aggregate_data = load_aggregate_data(run_id)
    evaluations_sample = load_evaluations_sample(run_id)
    
    # Render sections based on navigation
    if selected_section == "📊 Run Overview":
        render_run_overview(run_data, eval_stats)
    elif selected_section == "🤖 Model Comparison":
        render_model_comparison(model_performance)
    elif selected_section == "🔄 Transform Analysis":
        render_transform_analysis(aggregate_data)
    elif selected_section == "🔍 Evaluation Details":
        render_evaluation_details(evaluations_sample)
    else:
        # Default: show all sections
        render_run_overview(run_data, eval_stats)
        st.divider()
        render_model_comparison(model_performance)
        st.divider()
        render_transform_analysis(aggregate_data)
        st.divider()
        render_evaluation_details(evaluations_sample)


if __name__ == "__main__":
    main()