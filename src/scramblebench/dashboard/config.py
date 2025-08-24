"""
Dashboard configuration and utilities.

Provides configuration constants and utility functions for the Streamlit dashboard.
"""

from typing import Dict, Any

# Dashboard configuration
DASHBOARD_CONFIG = {
    'page_title': 'ScrambleBench Dashboard',
    'page_icon': '🔬',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Cache settings
CACHE_SETTINGS = {
    'resource_ttl': None,  # Resources cached indefinitely
    'data_ttl': 300,       # Data cached for 5 minutes
    'long_data_ttl': 600   # Long-term data cached for 10 minutes
}

# Chart colors and themes
CHART_COLORS = {
    'accuracy': 'viridis',
    'latency': 'reds', 
    'cost': 'blues',
    'robustness': 'RdYlBu_r'
}

# Metrics display configuration
METRICS_CONFIG = {
    'accuracy': {
        'format': '%.1f%%',
        'multiplier': 100
    },
    'latency': {
        'format': '%.0fms',
        'multiplier': 1
    },
    'cost': {
        'format': '$%.4f',
        'multiplier': 1
    },
    'duration': {
        'format': '%.0fs',
        'multiplier': 1
    }
}

# Section navigation
NAVIGATION_SECTIONS = [
    "📊 Run Overview",
    "🤖 Model Comparison", 
    "🔄 Transform Analysis",
    "🔍 Evaluation Details"
]

# Default limits for data loading
DEFAULT_LIMITS = {
    'evaluation_sample': 100,
    'recent_runs': 20,
    'top_performers': 10
}


def format_metric(value: float, metric_type: str) -> str:
    """Format a metric value according to its type."""
    if value is None:
        return "N/A"
    
    config = METRICS_CONFIG.get(metric_type, {'format': '%.2f', 'multiplier': 1})
    formatted_value = value * config['multiplier']
    return config['format'] % formatted_value


def get_status_emoji(status: str) -> str:
    """Get emoji for run status."""
    status_emojis = {
        'completed': '✅',
        'running': '🔄', 
        'failed': '❌',
        'cancelled': '⏹️'
    }
    return status_emojis.get(status.lower(), '❓')


def get_boolean_emoji(value: bool) -> str:
    """Get emoji for boolean values."""
    return '✅' if value else '❌'