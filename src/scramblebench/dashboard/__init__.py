"""
ScrambleBench Dashboard Module.

Interactive Streamlit dashboard for exploring ScrambleBench evaluation results.

Usage:
    streamlit run src/scramblebench/dashboard/app.py -- --run-id {{RUN_ID}}

Components:
    - app: Main Streamlit dashboard application
    - config: Dashboard configuration and utilities
"""

from . import config

__all__ = ['config']