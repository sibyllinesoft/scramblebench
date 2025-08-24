# ScrambleBench Dashboard

Interactive Streamlit dashboard for exploring and visualizing ScrambleBench evaluation results.

## Overview

The ScrambleBench Dashboard provides a comprehensive, interactive interface for analyzing completed evaluation runs. It connects directly to the ScrambleBench database via the ORM and presents key metrics, visualizations, and detailed analysis in an easy-to-use web interface.

## Features

### 📊 Run Overview
- Run metadata and progress tracking
- Overall performance metrics
- Configuration details and timeline
- Real-time status monitoring

### 🤖 Model Comparison
- Side-by-side model performance analysis
- Accuracy, latency, and cost comparisons
- Interactive charts and detailed tables
- Performance ranking and statistics

### 🔄 Transform Analysis  
- Robustness analysis across different transforms
- Heatmaps showing model-transform combinations
- RRS (Relative Robustness Score) analysis
- Language Dependency Coefficient (LDC) visualization

### 🔍 Evaluation Details
- Sample-level evaluation inspection
- Distribution analysis of latency and costs
- Filtering by model, transform, and results
- Detailed evaluation data tables

## Usage

### Command Line
```bash
# Launch dashboard for a specific run
streamlit run src/scramblebench/dashboard/app.py -- --run-id {{RUN_ID}}

# Launch dashboard with run selection
streamlit run src/scramblebench/dashboard/app.py
```

### From ScrambleBench CLI
```bash
# View results from a completed run
scramblebench dashboard {{RUN_ID}}

# List available runs and launch dashboard  
scramblebench dashboard --list
```

## Requirements

- **Database**: The dashboard requires access to a ScrambleBench database with completed evaluation runs
- **Dependencies**: Streamlit, Plotly, Pandas (automatically installed with ScrambleBench)
- **Data**: At least one completed evaluation run in the database

## Navigation

The dashboard provides multiple ways to navigate:

1. **Sidebar Navigation**: Jump directly to specific sections
2. **Run Selection**: Switch between different evaluation runs
3. **Interactive Filters**: Filter data within each section
4. **System Status**: Monitor database connectivity

## Data Caching

The dashboard implements intelligent caching to improve performance:

- **Resource Caching**: Database connections cached indefinitely
- **Data Caching**: Query results cached for 5 minutes
- **Manual Refresh**: Use "🔄 Refresh Data" button to clear cache

## Troubleshooting

### Database Connection Issues
- Verify the ScrambleBench database exists and is accessible
- Check database configuration in your environment
- Use the System Status panel in the sidebar for diagnostics

### No Data Available
- Ensure at least one evaluation run has completed successfully
- Verify the run_id exists in the database
- Check run status - only completed runs show full analysis

### Performance Issues
- Use the data refresh button to clear stale caches
- Consider limiting the number of evaluations loaded for large runs
- Check database performance if queries are slow

## Development

### Architecture
- **Frontend**: Streamlit with Plotly charts
- **Backend**: Direct database access via ScrambleBench ORM
- **Caching**: Multi-level caching for performance
- **Responsive**: Designed for desktop and tablet viewing

### Extending
- Add new sections by creating render functions
- Modify `NAVIGATION_SECTIONS` in `config.py`
- Add new chart types using Plotly Express
- Extend repository methods for new data queries

### Testing
```bash
# Test dashboard with sample data
python -m pytest tests/dashboard/

# Manual testing with development database  
SCRAMBLEBENCH_DB_URL="sqlite:///test.db" streamlit run src/scramblebench/dashboard/app.py
```