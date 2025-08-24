-- ==============================================================================
-- LANGUAGE DEPENDENCY ATLAS - EXAMPLE RESEARCH QUERIES
-- ==============================================================================
-- This file contains example queries for common research operations
-- Copy and modify these queries for your specific research needs
-- ==============================================================================

-- ==============================================================================
-- 1. LANGUAGE DEPENDENCY COEFFICIENT CALCULATIONS
-- ==============================================================================

-- Calculate language dependency coefficients across all model families
-- This is a key metric for the research: how much does performance depend on text structure?
SELECT 
    m.model_family,
    m.model_name,
    ta.scrambling_method,
    ta.dependency_coefficient,
    ta.coefficient_interpretation,
    ta.baseline_performance,
    ta.threshold_value,
    ta.is_significant,
    ta.p_value
FROM threshold_analyses ta
JOIN models m ON ta.model_id = m.model_id
JOIN experiments e ON ta.experiment_id = e.experiment_id
WHERE ta.is_significant = true
  AND e.experiment_name = 'YOUR_EXPERIMENT_NAME' -- Replace with your experiment
ORDER BY m.model_family, ta.dependency_coefficient DESC;

-- Compare dependency coefficients between model families
SELECT 
    m.model_family,
    COUNT(DISTINCT m.model_id) as models_tested,
    AVG(ta.dependency_coefficient) as avg_dependency,
    STDDEV(ta.dependency_coefficient) as dependency_std,
    MIN(ta.dependency_coefficient) as min_dependency,
    MAX(ta.dependency_coefficient) as max_dependency
FROM threshold_analyses ta
JOIN models m ON ta.model_id = m.model_id
WHERE ta.is_significant = true
  AND ta.scrambling_method = 'character' -- Focus on one method
GROUP BY m.model_family
HAVING COUNT(DISTINCT m.model_id) >= 2 -- Only families with multiple models
ORDER BY avg_dependency DESC;

-- ==============================================================================
-- 2. PERFORMANCE CURVE ANALYSIS
-- ==============================================================================

-- Generate data for plotting performance vs scrambling intensity
-- This creates the classic "performance degradation curve" for academic papers
SELECT 
    m.model_family,
    m.model_name,
    pm.scrambling_intensity,
    pm.accuracy,
    pm.confidence_interval_95_lower,
    pm.confidence_interval_95_upper,
    pm.total_responses as sample_size
FROM performance_metrics pm
JOIN models m ON pm.model_id = m.model_id
JOIN experiments e ON pm.experiment_id = e.experiment_id
WHERE e.experiment_name = 'YOUR_EXPERIMENT_NAME'
  AND pm.scrambling_method = 'character'
  AND pm.total_responses >= 10 -- Minimum sample size
ORDER BY m.model_family, m.model_name, pm.scrambling_intensity;

-- Compare performance curves between model families
-- Statistical comparison of how different model types handle scrambling
WITH family_performance AS (
    SELECT 
        m.model_family,
        pm.scrambling_intensity,
        AVG(pm.accuracy) as avg_accuracy,
        STDDEV(pm.accuracy) as accuracy_std,
        COUNT(DISTINCT pm.model_id) as models_in_family
    FROM performance_metrics pm
    JOIN models m ON pm.model_id = m.model_id
    WHERE pm.scrambling_method = 'character'
    GROUP BY m.model_family, pm.scrambling_intensity
    HAVING COUNT(DISTINCT pm.model_id) >= 2
)
SELECT 
    model_family,
    scrambling_intensity,
    avg_accuracy,
    accuracy_std,
    models_in_family,
    -- Calculate performance retention (compared to 0% scrambling)
    avg_accuracy / FIRST_VALUE(avg_accuracy) OVER (
        PARTITION BY model_family 
        ORDER BY scrambling_intensity
    ) as performance_retention
FROM family_performance
ORDER BY model_family, scrambling_intensity;

-- ==============================================================================
-- 3. THRESHOLD DETECTION ANALYSIS
-- ==============================================================================

-- Find critical scrambling thresholds where models break down
-- Answers: "At what scrambling intensity do models lose 50% of their performance?"
SELECT 
    m.model_family,
    m.model_name,
    ta.threshold_value,
    ta.baseline_performance,
    ta.threshold_performance,
    ta.performance_drop,
    ta.p_value,
    -- Calculate how robust the model is (higher threshold = more robust)
    CASE 
        WHEN ta.threshold_value >= 75 THEN 'Very Robust'
        WHEN ta.threshold_value >= 50 THEN 'Robust'
        WHEN ta.threshold_value >= 25 THEN 'Moderate'
        ELSE 'Fragile'
    END as robustness_category
FROM threshold_analyses ta
JOIN models m ON ta.model_id = m.model_id
WHERE ta.is_significant = true
  AND ta.scrambling_method = 'character'
ORDER BY ta.threshold_value DESC;

-- Statistical comparison of threshold values between model families
SELECT 
    m.model_family,
    COUNT(*) as models_tested,
    AVG(ta.threshold_value) as avg_threshold,
    STDDEV(ta.threshold_value) as threshold_std,
    MIN(ta.threshold_value) as min_threshold,
    MAX(ta.threshold_value) as max_threshold,
    -- Statistical significance test between families (requires additional analysis)
    AVG(ta.dependency_coefficient) as avg_dependency_coeff
FROM threshold_analyses ta
JOIN models m ON ta.model_id = m.model_id
WHERE ta.is_significant = true
  AND ta.scrambling_method = 'character'
GROUP BY m.model_family
ORDER BY avg_threshold DESC;

-- ==============================================================================
-- 4. DOMAIN-SPECIFIC PERFORMANCE ANALYSIS
-- ==============================================================================

-- How do different domains (reasoning, knowledge, etc.) respond to scrambling?
SELECT 
    dp.domain,
    m.model_family,
    AVG(dp.baseline_accuracy) as avg_baseline,
    AVG(dp.extreme_scrambling_accuracy) as avg_extreme_scrambled,
    AVG(dp.robustness_score) as avg_robustness,
    AVG(dp.degradation_rate) as avg_degradation_rate,
    COUNT(DISTINCT dp.model_id) as models_tested
FROM domain_performance dp
JOIN models m ON dp.model_id = m.model_id
GROUP BY dp.domain, m.model_family
HAVING COUNT(DISTINCT dp.model_id) >= 2
ORDER BY dp.domain, avg_robustness DESC;

-- Find domains where scrambling has the biggest impact
SELECT 
    domain,
    AVG(baseline_accuracy - extreme_scrambling_accuracy) as avg_performance_drop,
    AVG(degradation_rate) as avg_degradation_rate,
    COUNT(DISTINCT model_id) as models_tested,
    -- Rank domains by how much scrambling hurts performance
    RANK() OVER (ORDER BY AVG(degradation_rate) DESC) as impact_rank
FROM domain_performance
GROUP BY domain
HAVING COUNT(DISTINCT model_id) >= 3
ORDER BY avg_performance_drop DESC;

-- ==============================================================================
-- 5. COGNITIVE LOAD ANALYSIS
-- ==============================================================================

-- How does cognitive load interact with scrambling effects?
-- This helps answer whether complex questions are more susceptible to scrambling
SELECT 
    cl.load_category,
    pm.scrambling_intensity,
    AVG(pm.accuracy) as avg_accuracy,
    STDDEV(pm.accuracy) as accuracy_std,
    COUNT(DISTINCT pm.model_id) as models_tested,
    COUNT(*) as total_measurements
FROM cognitive_load_classifications cl
JOIN questions q ON cl.question_id = q.question_id
JOIN responses r ON q.question_id = r.original_question_id
JOIN performance_metrics pm ON r.experiment_id = pm.experiment_id AND r.model_id = pm.model_id
WHERE pm.scrambling_method = 'character'
  AND r.is_correct IS NOT NULL
GROUP BY cl.load_category, pm.scrambling_intensity
HAVING COUNT(*) >= 5 -- Minimum sample size
ORDER BY cl.load_category, pm.scrambling_intensity;

-- Correlation between cognitive load and scrambling susceptibility
WITH load_scrambling_correlation AS (
    SELECT 
        cl.question_id,
        cl.overall_cognitive_load,
        -- Calculate average accuracy drop across all scrambling levels for this question
        AVG(CASE WHEN sq.scrambling_intensity = 0 THEN 1.0 
                 WHEN r.is_correct THEN 1.0 ELSE 0.0 END) 
        - AVG(CASE WHEN sq.scrambling_intensity > 50 THEN 
                   CASE WHEN r.is_correct THEN 1.0 ELSE 0.0 END END) as performance_drop
    FROM cognitive_load_classifications cl
    JOIN questions q ON cl.question_id = q.question_id
    JOIN responses r ON q.question_id = r.original_question_id OR q.question_id IN (
        SELECT sq2.original_question_id FROM scrambled_questions sq2 WHERE sq2.scrambled_id = r.scrambled_question_id
    )
    LEFT JOIN scrambled_questions sq ON r.scrambled_question_id = sq.scrambled_id
    WHERE r.is_correct IS NOT NULL
    GROUP BY cl.question_id, cl.overall_cognitive_load
    HAVING COUNT(*) >= 5
)
SELECT 
    -- Correlation analysis would typically be done in statistical software
    -- This gives you the raw data for that analysis
    overall_cognitive_load,
    performance_drop,
    COUNT(*) as question_count
FROM load_scrambling_correlation
GROUP BY overall_cognitive_load, performance_drop
ORDER BY overall_cognitive_load;

-- ==============================================================================
-- 6. MODEL COMPARISON AND RANKING
-- ==============================================================================

-- Comprehensive model comparison for academic papers
SELECT 
    m.model_family,
    m.model_name,
    m.parameter_count,
    
    -- Overall performance metrics
    AVG(CASE WHEN pm.scrambling_intensity = 0 THEN pm.accuracy END) as baseline_accuracy,
    AVG(pm.accuracy) as overall_avg_accuracy,
    
    -- Robustness metrics  
    MAX(ta.dependency_coefficient) as max_dependency_coeff,
    AVG(ta.threshold_value) as avg_threshold,
    
    -- Economic metrics
    SUM(r.cost) as total_experiment_cost,
    AVG(r.response_time_ms) as avg_response_time,
    
    -- Sample size
    COUNT(DISTINCT r.response_id) as total_responses,
    COUNT(DISTINCT pm.scrambling_intensity) as scrambling_levels_tested
    
FROM models m
JOIN responses r ON m.model_id = r.model_id
JOIN performance_metrics pm ON r.experiment_id = pm.experiment_id AND r.model_id = pm.model_id
LEFT JOIN threshold_analyses ta ON pm.experiment_id = ta.experiment_id AND pm.model_id = ta.model_id
WHERE r.is_correct IS NOT NULL
GROUP BY m.model_id, m.model_family, m.model_name, m.parameter_count
HAVING COUNT(DISTINCT r.response_id) >= 100 -- Minimum responses for reliable comparison
ORDER BY baseline_accuracy DESC, avg_threshold DESC;

-- Model performance ranking with statistical significance
-- This helps determine which models are statistically significantly better
WITH model_stats AS (
    SELECT 
        m.model_id,
        m.model_name,
        AVG(pm.accuracy) as avg_accuracy,
        STDDEV(pm.accuracy) as accuracy_std,
        COUNT(*) as n_measurements
    FROM models m
    JOIN performance_metrics pm ON m.model_id = pm.model_id
    WHERE pm.scrambling_method = 'character'
    GROUP BY m.model_id, m.model_name
    HAVING COUNT(*) >= 5
)
SELECT 
    m1.model_name as model_1,
    m2.model_name as model_2,
    m1.avg_accuracy as accuracy_1,
    m2.avg_accuracy as accuracy_2,
    ABS(m1.avg_accuracy - m2.avg_accuracy) as accuracy_difference,
    -- For t-test calculation (would need statistical software for p-values)
    SQRT((m1.accuracy_std^2/m1.n_measurements) + (m2.accuracy_std^2/m2.n_measurements)) as pooled_se
FROM model_stats m1
CROSS JOIN model_stats m2
WHERE m1.model_id < m2.model_id -- Avoid duplicate comparisons
  AND ABS(m1.avg_accuracy - m2.avg_accuracy) > 0.05 -- Only meaningful differences
ORDER BY accuracy_difference DESC;

-- ==============================================================================
-- 7. REPRODUCIBILITY AND EXPERIMENT VALIDATION
-- ==============================================================================

-- Check experiment reproducibility
SELECT 
    e.experiment_name,
    e.git_commit_hash,
    e.random_seed,
    e.environment_snapshot->>'python_version' as python_version,
    COUNT(DISTINCT r.model_id) as models_tested,
    COUNT(DISTINCT r.response_id) as total_responses,
    MIN(r.created_at) as first_response,
    MAX(r.created_at) as last_response,
    e.status
FROM experiments e
LEFT JOIN responses r ON e.experiment_id = r.experiment_id
GROUP BY e.experiment_id, e.experiment_name, e.git_commit_hash, e.random_seed, 
         e.environment_snapshot, e.status
ORDER BY e.created_at DESC;

-- Validate data quality and completeness
SELECT 
    e.experiment_name,
    COUNT(DISTINCT m.model_id) as unique_models,
    COUNT(DISTINCT q.question_id) as unique_questions,
    COUNT(DISTINCT sq.scrambling_intensity) as scrambling_levels,
    COUNT(r.response_id) as total_responses,
    COUNT(r.response_id) FILTER (WHERE r.is_correct IS NOT NULL) as evaluated_responses,
    ROUND(100.0 * COUNT(r.response_id) FILTER (WHERE r.is_correct IS NOT NULL) / COUNT(r.response_id), 2) as evaluation_completeness_pct,
    COUNT(pm.metric_id) as computed_metrics
FROM experiments e
LEFT JOIN responses r ON e.experiment_id = r.experiment_id
LEFT JOIN models m ON r.model_id = m.model_id
LEFT JOIN questions q ON r.original_question_id = q.question_id
LEFT JOIN scrambled_questions sq ON r.scrambled_question_id = sq.scrambled_id
LEFT JOIN performance_metrics pm ON e.experiment_id = pm.experiment_id
GROUP BY e.experiment_id, e.experiment_name
ORDER BY total_responses DESC;

-- ==============================================================================
-- 8. EXPORT QUERIES FOR EXTERNAL ANALYSIS
-- ==============================================================================

-- Export data for R/Python statistical analysis
-- This creates a comprehensive dataset with all variables needed for analysis
SELECT 
    e.experiment_name,
    m.model_family,
    m.model_name,
    m.parameter_count,
    COALESCE(sq.scrambling_method, 'original') as scrambling_method,
    COALESCE(sq.scrambling_intensity, 0.0) as scrambling_intensity,
    q_final.domain,
    q_final.question_type,
    q_final.difficulty_rating,
    cl.overall_cognitive_load,
    cl.load_category,
    CASE WHEN r.is_correct THEN 1 ELSE 0 END as accuracy,
    r.partial_credit,
    r.response_time_ms,
    r.confidence_score,
    -- Additional derived variables for analysis
    CASE WHEN COALESCE(sq.scrambling_intensity, 0.0) = 0 THEN 'baseline' ELSE 'scrambled' END as condition,
    CASE 
        WHEN COALESCE(sq.scrambling_intensity, 0.0) = 0 THEN 'baseline'
        WHEN sq.scrambling_intensity <= 25 THEN 'low'
        WHEN sq.scrambling_intensity <= 50 THEN 'medium'
        WHEN sq.scrambling_intensity <= 75 THEN 'high'
        ELSE 'extreme'
    END as scrambling_category
FROM responses r
JOIN experiments e ON r.experiment_id = e.experiment_id
JOIN models m ON r.model_id = m.model_id
LEFT JOIN questions q ON r.original_question_id = q.question_id
LEFT JOIN scrambled_questions sq ON r.scrambled_question_id = sq.scrambled_id
LEFT JOIN questions q2 ON sq.original_question_id = q2.question_id
LEFT JOIN cognitive_load_classifications cl ON COALESCE(q.question_id, q2.question_id) = cl.question_id
-- Use COALESCE to get question info from either original or scrambled
LEFT JOIN questions q_final ON COALESCE(q.question_id, q2.question_id) = q_final.question_id
WHERE r.is_correct IS NOT NULL
  AND e.experiment_name = 'YOUR_EXPERIMENT_NAME' -- Replace with your experiment
ORDER BY m.model_family, m.model_name, scrambling_intensity;

-- Export summary statistics for quick analysis
SELECT 
    'Model Performance Summary' as analysis_type,
    m.model_family,
    m.model_name,
    COUNT(*) as total_responses,
    AVG(CASE WHEN r.is_correct THEN 1.0 ELSE 0.0 END) as overall_accuracy,
    AVG(CASE WHEN COALESCE(sq.scrambling_intensity, 0) = 0 AND r.is_correct THEN 1.0 
             WHEN COALESCE(sq.scrambling_intensity, 0) = 0 THEN 0.0 END) as baseline_accuracy,
    AVG(CASE WHEN sq.scrambling_intensity > 50 AND r.is_correct THEN 1.0 
             WHEN sq.scrambling_intensity > 50 THEN 0.0 END) as high_scrambling_accuracy,
    AVG(r.response_time_ms) as avg_response_time
FROM responses r
JOIN models m ON r.model_id = m.model_id
LEFT JOIN scrambled_questions sq ON r.scrambled_question_id = sq.scrambled_id
WHERE r.is_correct IS NOT NULL
GROUP BY m.model_family, m.model_name
HAVING COUNT(*) >= 50 -- Minimum sample size
ORDER BY overall_accuracy DESC;

-- ==============================================================================
-- 9. ADVANCED STATISTICAL QUERIES
-- ==============================================================================

-- Analysis of variance (ANOVA) setup data
-- Prepares data for ANOVA testing: Does scrambling method significantly affect performance?
SELECT 
    r.response_id,
    m.model_family as group_factor,
    COALESCE(sq.scrambling_method, 'original') as treatment,
    CASE WHEN r.is_correct THEN 1.0 ELSE 0.0 END as dependent_variable,
    COALESCE(sq.scrambling_intensity, 0.0) as covariate
FROM responses r
JOIN models m ON r.model_id = m.model_id
LEFT JOIN scrambled_questions sq ON r.scrambled_question_id = sq.scrambled_id
WHERE r.is_correct IS NOT NULL
  AND m.model_family IN ('gpt', 'llama', 'claude') -- Focus on major families
ORDER BY group_factor, treatment;

-- Correlation analysis setup
-- Examines relationships between model characteristics and scrambling sensitivity
WITH model_characteristics AS (
    SELECT 
        m.model_id,
        m.model_family,
        m.parameter_count,
        AVG(CASE WHEN r.is_correct THEN 1.0 ELSE 0.0 END) as overall_accuracy,
        -- Calculate scrambling sensitivity (baseline - scrambled performance)
        AVG(CASE WHEN COALESCE(sq.scrambling_intensity, 0) = 0 AND r.is_correct THEN 1.0 
                 WHEN COALESCE(sq.scrambling_intensity, 0) = 0 THEN 0.0 END) -
        AVG(CASE WHEN sq.scrambling_intensity > 0 AND r.is_correct THEN 1.0 
                 WHEN sq.scrambling_intensity > 0 THEN 0.0 END) as scrambling_sensitivity,
        AVG(r.response_time_ms) as avg_response_time
    FROM models m
    JOIN responses r ON m.model_id = r.model_id
    LEFT JOIN scrambled_questions sq ON r.scrambled_question_id = sq.scrambled_id
    WHERE r.is_correct IS NOT NULL
    GROUP BY m.model_id, m.model_family, m.parameter_count
    HAVING COUNT(*) >= 20 -- Minimum sample size
)
SELECT 
    model_family,
    parameter_count,
    overall_accuracy,
    scrambling_sensitivity,
    avg_response_time
FROM model_characteristics
WHERE parameter_count IS NOT NULL
ORDER BY parameter_count;

-- ==============================================================================
-- USAGE NOTES
-- ==============================================================================
/*
TO USE THESE QUERIES:

1. Replace 'YOUR_EXPERIMENT_NAME' with your actual experiment name
2. Adjust minimum sample sizes based on your data volume
3. Modify scrambling_method filters ('character', 'word', etc.) as needed
4. For statistical significance testing, export data and use R, Python, or SPSS
5. Consider creating views from frequently used queries for better performance

PERFORMANCE TIPS:
- Add WHERE clauses to limit data when possible
- Use EXPLAIN ANALYZE to check query performance
- Consider adding indexes on frequently filtered columns
- For large datasets, consider materialized views for complex aggregations

STATISTICAL ANALYSIS WORKFLOW:
1. Use these queries to extract raw data
2. Export to CSV using the export functions
3. Import into statistical software (R, Python pandas, SPSS)
4. Perform formal statistical tests (ANOVA, t-tests, regression)
5. Generate publication-quality figures and tables
6. Store results back in statistical_analyses table for reproducibility
*/