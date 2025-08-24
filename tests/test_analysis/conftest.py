"""Pytest fixtures and configuration for analysis module tests."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock


@pytest.fixture
def sample_benchmark_results():
    """Fixture providing sample benchmark results for testing."""
    return {
        'model_a': {
            'scores': [0.85, 0.87, 0.83, 0.89, 0.86],
            'response_times': [120, 135, 110, 145, 125],
            'metadata': {'params': '7B', 'type': 'instruction_following'}
        },
        'model_b': {
            'scores': [0.78, 0.80, 0.76, 0.82, 0.79],
            'response_times': [90, 95, 85, 100, 88],
            'metadata': {'params': '13B', 'type': 'chat'}
        },
        'model_c': {
            'scores': [0.92, 0.94, 0.90, 0.95, 0.93],
            'response_times': [200, 210, 195, 220, 205],
            'metadata': {'params': '70B', 'type': 'instruction_following'}
        }
    }


@pytest.fixture
def sample_dataframe(sample_benchmark_results):
    """Fixture providing sample benchmark data as pandas DataFrame."""
    rows = []
    for model_name, data in sample_benchmark_results.items():
        for i, (score, time) in enumerate(zip(data['scores'], data['response_times'])):
            rows.append({
                'model': model_name,
                'score': score,
                'response_time': time,
                'run_id': i,
                'params': data['metadata']['params'],
                'type': data['metadata']['type']
            })
    return pd.DataFrame(rows)


@pytest.fixture
def temp_output_dir():
    """Fixture providing a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_statistical_model():
    """Fixture providing a mock statistical model."""
    model = Mock()
    model.fit.return_value = model
    model.predict.return_value = np.array([0.85, 0.90, 0.78])
    model.confidence_interval.return_value = (np.array([0.80, 0.85, 0.73]), np.array([0.90, 0.95, 0.83]))
    return model


@pytest.fixture
def mock_bootstrap_sampler():
    """Fixture providing a mock bootstrap sampler."""
    sampler = Mock()
    sampler.sample.return_value = np.array([
        [0.84, 0.86, 0.88],
        [0.83, 0.87, 0.85],
        [0.85, 0.89, 0.87]
    ])
    return sampler


@pytest.fixture(autouse=True)
def setup_matplotlib_backend():
    """Automatically set matplotlib to use Agg backend for testing."""
    import matplotlib
    matplotlib.use('Agg')
    yield
    matplotlib.pyplot.close('all')


@pytest.fixture
def sample_analysis_config():
    """Fixture providing sample analysis configuration."""
    return {
        'bootstrap': {
            'n_samples': 1000,
            'confidence_level': 0.95,
            'method': 'percentile'
        },
        'visualization': {
            'figure_size': (10, 6),
            'dpi': 300,
            'format': 'png'
        },
        'export': {
            'latex_format': 'booktabs',
            'decimal_places': 3
        }
    }