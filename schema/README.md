# Language Dependency Atlas Database Schema

## Overview

This database schema is designed specifically for academic research on language dependency in AI models using scrambled text experiments. It supports comprehensive data collection, statistical analysis, and academic publication workflows for the ScrambleBench project.

## 🎯 Research Goals Supported

- **Language Dependency Measurement**: Calculate how much model performance depends on text structure
- **Threshold Detection**: Identify critical scrambling levels where models fail
- **Model Comparison**: Compare robustness across different model families and sizes  
- **Domain Analysis**: Understand how different question types respond to scrambling
- **Cognitive Load Studies**: Analyze interaction between question complexity and scrambling effects
- **Reproducibility**: Full experimental reproducibility and replication support

## 📊 Key Database Features

### Core Experimental Data
- **Experiments**: Complete metadata, reproducibility info, and git tracking
- **Models**: Technical specifications, API details, and performance characteristics
- **Questions**: Original test items with linguistic features and cognitive load classifications
- **Scrambled Questions**: Transformed versions with detailed transformation logs
- **Responses**: Model outputs with evaluation results and performance metrics

### Research Analytics
- **Performance Metrics**: Pre-computed aggregations for fast statistical queries
- **Threshold Analyses**: Language dependency coefficients and critical points
- **Statistical Analyses**: Results from ANOVA, t-tests, and correlation studies
- **Domain Performance**: Specialized breakdowns by question domain/type
- **Visualization Data**: Pre-formatted data for academic figures and charts

### Academic Features
- **Full Reproducibility**: Git commits, environment snapshots, and random seeds
- **Statistical Export**: CSV/R/Python export functions for external analysis
- **Quality Assurance**: Data validation triggers and integrity constraints
- **Schema Versioning**: Evolution tracking for longitudinal studies
- **Performance Optimization**: Indexes optimized for research query patterns

## 🚀 Quick Start

### 1. Database Setup

```bash
# Create PostgreSQL database
createdb language_dependency_atlas

# Apply the schema
psql language_dependency_atlas -f language_dependency_atlas.sql
```

### 2. Basic Data Loading

```sql
-- Create a sample experiment
SELECT create_sample_experiment(
    'Language Dependency Baseline Study',
    'Initial experiment measuring language dependency across major model families',
    'researcher_name'
);

-- Add models
INSERT INTO models (model_name, model_family, model_version, parameter_count, access_type)
VALUES 
    ('gpt-3.5-turbo', 'gpt', '3.5', 175000000000, 'api'),
    ('llama-2-7b', 'llama', '2-7b', 7000000000, 'local'),
    ('claude-3-haiku', 'claude', '3-haiku', NULL, 'api');

-- Add a benchmark
INSERT INTO benchmarks (benchmark_name, description, domain, question_count, created_by)
VALUES ('MMLU-Sample', 'Sample questions from MMLU benchmark', 'knowledge', 100, 'researcher');
```

### 3. Example Research Queries

```sql
-- Calculate language dependency coefficients
SELECT * FROM language_dependency_summary 
WHERE experiment_name = 'Language Dependency Baseline Study';

-- Generate performance curves for visualization
SELECT * FROM performance_curves 
WHERE scrambling_method = 'character'
ORDER BY model_name, scrambling_intensity;

-- Export data for statistical analysis
SELECT export_experiment_data('your-experiment-id', false);
```

## 📈 Key Research Metrics

### Language Dependency Coefficient
- **Range**: 0.0 (no dependency) to 1.0 (complete dependency)
- **Calculation**: `1 - (average_scrambled_performance / baseline_performance)`
- **Interpretation**: Higher values = more dependent on text structure

### Threshold Values
- **Definition**: Scrambling intensity where performance drops to 50% of baseline
- **Usage**: Compare model robustness (higher threshold = more robust)
- **Statistical Testing**: Includes confidence intervals and significance tests

### Performance Retention
- **Formula**: `scrambled_accuracy / baseline_accuracy`  
- **Analysis**: Track performance degradation across scrambling levels
- **Visualization**: Creates characteristic "performance curves" for academic papers

## 🔍 Common Research Workflows

### 1. Model Comparison Study
```sql
-- Compare language dependency across model families
SELECT 
    model_family,
    AVG(dependency_coefficient) as avg_dependency,
    COUNT(*) as models_tested
FROM language_dependency_summary
GROUP BY model_family
ORDER BY avg_dependency DESC;
```

### 2. Domain-Specific Analysis  
```sql
-- Analyze which domains are most affected by scrambling
SELECT 
    domain,
    AVG(baseline_accuracy - extreme_scrambling_accuracy) as performance_drop
FROM domain_performance
GROUP BY domain
ORDER BY performance_drop DESC;
```

### 3. Threshold Detection
```sql
-- Find models with highest scrambling thresholds (most robust)
SELECT 
    model_name,
    threshold_value,
    robustness_category
FROM threshold_analyses ta
JOIN models m ON ta.model_id = m.model_id  
WHERE is_significant = true
ORDER BY threshold_value DESC;
```

## 📊 Database Performance

### Optimized Query Patterns
- **Model comparisons**: Indexed on `(model_id, experiment_id)`
- **Scrambling analysis**: Indexed on `(scrambling_method, scrambling_intensity)`
- **Statistical queries**: Composite indexes for multi-dimensional analysis
- **Time series**: Partitioned by experiment date for large datasets

### Performance Metrics
- **Response storage**: ~1KB per response (including metadata)
- **Query speed**: Sub-second for most analytical queries with proper indexes
- **Scalability**: Supports millions of responses across hundreds of experiments
- **Export speed**: ~1000 responses/second for CSV generation

## 🔧 Advanced Features

### Statistical Analysis Integration
```sql
-- ANOVA setup data
SELECT model_family, scrambling_method, accuracy 
FROM responses r
JOIN models m ON r.model_id = m.model_id
-- Export to R/Python for formal statistical testing
```

### Visualization Support
```sql
-- Pre-formatted data for academic figures
SELECT * FROM visualization_data 
WHERE chart_type = 'performance_curve'
  AND experiment_id = 'your-experiment-id';
```

### Reproducibility Tracking
```sql
-- Complete experiment reproducibility info
SELECT * FROM experiment_reproducibility
WHERE experiment_name = 'your-experiment';
```

## 📚 File Structure

```
schema/
├── language_dependency_atlas.sql    # Main schema definition
├── example_queries.sql              # Research query examples  
├── README.md                       # This documentation
└── migrations/                     # Schema evolution scripts
    ├── v1.0.0_initial.sql
    ├── v1.1.0_add_cognitive_load.sql
    └── v1.2.0_add_visualization.sql
```

## 🔄 Schema Evolution

The database supports versioned schema evolution for longitudinal research:

```sql
-- Check current schema version
SELECT * FROM schema_versions ORDER BY applied_at DESC LIMIT 1;

-- Apply migrations
-- psql database -f migrations/v1.1.0_add_cognitive_load.sql
```

## 💾 Data Export Options

### For Statistical Analysis
```sql
-- Full dataset export
SELECT export_experiment_data('experiment-id', false);

-- Summary statistics only  
SELECT * FROM model_comparison_summary;
```

### For Visualization
```sql
-- Performance curves
SELECT * FROM performance_curves WHERE experiment_name = 'your-study';

-- Threshold comparisons
SELECT * FROM language_dependency_summary;
```

### For Replication
```sql
-- Complete reproducibility data
SELECT * FROM experiment_reproducibility;

-- Configuration and environment info
SELECT configuration, environment_snapshot 
FROM experiments WHERE experiment_name = 'your-study';
```

## 🎓 Academic Publication Support

### Tables and Figures
- **Table 1**: Model comparison summary (`model_comparison_summary` view)
- **Figure 1**: Performance curves (`performance_curves` view)  
- **Figure 2**: Threshold comparison (`language_dependency_summary` view)
- **Table 2**: Statistical analysis results (`statistical_analyses` table)

### Reproducibility Information
- Git commit hashes and environment snapshots stored with every experiment
- Complete parameter configurations preserved  
- Random seeds tracked for deterministic replication
- Hardware specifications recorded for performance analysis

### Data Sharing
- CSV exports formatted for other researchers
- R/Python code generation for analysis replication
- Standardized metadata following research data management best practices

## ⚠️ Important Notes

### Data Integrity
- Foreign key constraints ensure referential integrity
- Check constraints validate data ranges and business rules
- Triggers prevent common data entry errors
- Unique constraints prevent duplicate responses

### Performance Considerations  
- Index maintenance may slow bulk inserts
- Large exports should be run during off-peak hours
- Consider table partitioning for multi-year studies
- Monitor query performance with `EXPLAIN ANALYZE`

### Security and Ethics
- No personal data stored (only model responses and metadata)
- Institutional approval tracking built into experiments table  
- Funding source documentation for transparency
- License compliance tracking for benchmark data

## 📞 Support and Contribution

For questions about the schema design or research applications:
- Review the example queries in `example_queries.sql`
- Check the database health with `SELECT * FROM database_health`  
- Submit issues for schema improvements or additional research features
- Follow academic data management best practices for your institution

---

**Note**: This schema is optimized for academic research. For production applications, consider additional security, monitoring, and backup strategies appropriate for your environment.