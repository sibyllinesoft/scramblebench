-- ==============================================================================
-- Language Dependency Atlas Research Database Schema
-- ==============================================================================
-- Purpose: Comprehensive database design for academic research on language 
--          dependency in AI models using scrambled text experiments
-- Features: Reproducibility tracking, statistical analysis support, 
--           academic publication data export, performance optimization
-- ==============================================================================

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- ==============================================================================
-- CORE EXPERIMENTAL TABLES
-- ==============================================================================

-- Schema versioning for research evolution
CREATE TABLE schema_versions (
    version_id SERIAL PRIMARY KEY,
    version_number VARCHAR(20) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    migration_script TEXT,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    applied_by VARCHAR(100) NOT NULL
);

-- Insert initial schema version
INSERT INTO schema_versions (version_number, description, applied_by) 
VALUES ('1.0.0', 'Initial Language Dependency Atlas schema', 'system');

-- Research experiments (top-level container)
CREATE TABLE experiments (
    experiment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    research_question TEXT NOT NULL,
    hypothesis TEXT,
    
    -- Reproducibility metadata
    git_commit_hash VARCHAR(40) NOT NULL,
    git_branch VARCHAR(100) NOT NULL,
    environment_snapshot JSONB NOT NULL, -- Python version, libraries, etc.
    random_seed BIGINT,
    hardware_info JSONB,
    
    -- Timing and status
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) NOT NULL DEFAULT 'planned' 
        CHECK (status IN ('planned', 'running', 'completed', 'failed', 'cancelled')),
    
    -- Research metadata
    researcher_name VARCHAR(100) NOT NULL,
    institution VARCHAR(200),
    funding_source VARCHAR(200),
    ethical_approval_id VARCHAR(100),
    
    -- Configuration
    configuration JSONB NOT NULL, -- All experiment parameters
    notes TEXT,
    
    CONSTRAINT valid_timing CHECK (
        (started_at IS NULL OR started_at >= created_at) AND
        (completed_at IS NULL OR completed_at >= started_at)
    )
);

-- Language models being tested
CREATE TABLE models (
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL UNIQUE,
    model_family VARCHAR(50) NOT NULL, -- e.g., 'llama', 'gpt', 'claude'
    model_version VARCHAR(50) NOT NULL,
    architecture VARCHAR(50), -- e.g., 'transformer', 'lstm'
    
    -- Technical specifications
    parameter_count BIGINT, -- Number of parameters
    context_length INTEGER, -- Maximum context window
    training_cutoff DATE, -- Knowledge cutoff date
    
    -- API/Access details
    api_provider VARCHAR(50), -- 'openai', 'anthropic', 'ollama', etc.
    api_endpoint VARCHAR(255),
    access_type VARCHAR(20) NOT NULL CHECK (access_type IN ('api', 'local', 'hosted')),
    
    -- Model characteristics
    is_instruction_tuned BOOLEAN NOT NULL DEFAULT false,
    is_chat_model BOOLEAN NOT NULL DEFAULT false,
    supports_system_messages BOOLEAN NOT NULL DEFAULT false,
    
    -- Metadata
    description TEXT,
    paper_reference TEXT, -- Citation for model paper
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Performance characteristics
    typical_tokens_per_second DECIMAL(10,2),
    cost_per_1k_tokens DECIMAL(10,6),
    
    UNIQUE (model_name, model_version)
);

-- Test suites/benchmarks
CREATE TABLE benchmarks (
    benchmark_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    benchmark_name VARCHAR(100) NOT NULL UNIQUE,
    benchmark_version VARCHAR(20) NOT NULL DEFAULT '1.0',
    description TEXT NOT NULL,
    
    -- Benchmark characteristics
    domain VARCHAR(50) NOT NULL, -- 'reasoning', 'knowledge', 'comprehension', etc.
    difficulty_level VARCHAR(20) CHECK (difficulty_level IN ('easy', 'medium', 'hard', 'expert')),
    question_count INTEGER NOT NULL CHECK (question_count > 0),
    
    -- Source information
    source_paper TEXT, -- Citation
    source_url TEXT,
    license VARCHAR(50),
    
    -- Quality metrics
    inter_annotator_agreement DECIMAL(4,3), -- Kappa score or similar
    validation_method TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) NOT NULL,
    
    UNIQUE (benchmark_name, benchmark_version)
);

-- Individual questions/test items
CREATE TABLE questions (
    question_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    benchmark_id UUID NOT NULL REFERENCES benchmarks(benchmark_id) ON DELETE CASCADE,
    
    -- Question content
    question_text TEXT NOT NULL,
    question_type VARCHAR(50) NOT NULL, -- 'multiple_choice', 'open_ended', 'true_false'
    context TEXT, -- Additional context if needed
    
    -- Answer information
    correct_answer TEXT NOT NULL,
    answer_choices JSONB, -- For multiple choice: ["A", "B", "C", "D"]
    explanation TEXT, -- Why this is the correct answer
    
    -- Question metadata
    domain VARCHAR(50) NOT NULL,
    subdomain VARCHAR(50),
    difficulty_rating DECIMAL(3,2) CHECK (difficulty_rating BETWEEN 1.0 AND 5.0),
    cognitive_load_category VARCHAR(50), -- Research-specific classification
    
    -- Linguistic features
    word_count INTEGER NOT NULL,
    sentence_count INTEGER NOT NULL,
    avg_word_length DECIMAL(4,2),
    readability_score DECIMAL(4,2), -- Flesch-Kincaid or similar
    
    -- Quality assurance
    is_validated BOOLEAN NOT NULL DEFAULT false,
    validation_notes TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT positive_counts CHECK (word_count > 0 AND sentence_count > 0)
);

-- Scrambled versions of questions (key to language dependency research)
CREATE TABLE scrambled_questions (
    scrambled_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    original_question_id UUID NOT NULL REFERENCES questions(question_id) ON DELETE CASCADE,
    
    -- Scrambling parameters
    scrambling_method VARCHAR(50) NOT NULL, -- 'character', 'word', 'sentence', 'mixed'
    scrambling_intensity DECIMAL(5,2) NOT NULL CHECK (scrambling_intensity BETWEEN 0.0 AND 100.0),
    scrambling_seed INTEGER, -- For reproducible scrambling
    
    -- Scrambled content
    scrambled_text TEXT NOT NULL,
    scrambled_context TEXT, -- If context was also scrambled
    
    -- Scrambling details
    transformation_log JSONB NOT NULL, -- Detailed record of what changed
    preservation_rules JSONB, -- What elements were preserved (e.g., numbers, proper nouns)
    
    -- Quality metrics
    readability_change DECIMAL(5,2), -- How much readability was affected
    semantic_similarity DECIMAL(4,3), -- To original (if measurable)
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure we don't duplicate scrambling configurations
    UNIQUE (original_question_id, scrambling_method, scrambling_intensity, scrambling_seed)
);

-- Model responses to questions (original or scrambled)
CREATE TABLE responses (
    response_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    model_id UUID NOT NULL REFERENCES models(model_id),
    
    -- Question being answered (exactly one must be non-null)
    original_question_id UUID REFERENCES questions(question_id),
    scrambled_question_id UUID REFERENCES scrambled_questions(scrambled_id),
    
    -- Response content
    raw_response TEXT NOT NULL,
    processed_response TEXT, -- Cleaned/extracted answer
    confidence_score DECIMAL(4,3), -- If model provides confidence
    
    -- Performance metrics
    response_time_ms INTEGER NOT NULL CHECK (response_time_ms >= 0),
    token_count INTEGER,
    cost DECIMAL(10,6), -- Cost of this API call
    
    -- Evaluation results
    is_correct BOOLEAN,
    partial_credit DECIMAL(4,3) CHECK (partial_credit BETWEEN 0.0 AND 1.0),
    evaluation_method VARCHAR(50), -- 'exact_match', 'semantic_similarity', 'human_eval'
    human_evaluation_notes TEXT,
    
    -- Technical details
    api_request_id VARCHAR(255), -- For debugging API issues
    temperature DECIMAL(3,2),
    max_tokens INTEGER,
    prompt_template_id UUID, -- If using standardized prompts
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure exactly one question reference
    CONSTRAINT single_question_ref CHECK (
        (original_question_id IS NOT NULL)::int + (scrambled_question_id IS NOT NULL)::int = 1
    ),
    
    -- Unique constraint to prevent duplicate responses
    UNIQUE (experiment_id, model_id, original_question_id, scrambled_question_id)
);

-- ==============================================================================
-- ANALYSIS & RESULTS TABLES
-- ==============================================================================

-- Pre-computed performance metrics for fast querying
CREATE TABLE performance_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    model_id UUID NOT NULL REFERENCES models(model_id),
    benchmark_id UUID REFERENCES benchmarks(benchmark_id),
    
    -- Aggregation dimensions
    scrambling_method VARCHAR(50),
    scrambling_intensity DECIMAL(5,2),
    domain VARCHAR(50),
    question_type VARCHAR(50),
    
    -- Sample size
    total_responses INTEGER NOT NULL CHECK (total_responses > 0),
    valid_responses INTEGER NOT NULL CHECK (valid_responses <= total_responses),
    
    -- Performance metrics
    accuracy DECIMAL(6,5) NOT NULL CHECK (accuracy BETWEEN 0.0 AND 1.0),
    partial_credit_avg DECIMAL(6,5) CHECK (partial_credit_avg BETWEEN 0.0 AND 1.0),
    
    -- Response quality metrics
    avg_response_time_ms DECIMAL(8,2) NOT NULL,
    median_response_time_ms INTEGER NOT NULL,
    std_response_time_ms DECIMAL(8,2),
    
    -- Confidence metrics (if available)
    avg_confidence DECIMAL(4,3),
    confidence_calibration DECIMAL(4,3), -- How well confidence matches accuracy
    
    -- Statistical measures
    confidence_interval_95_lower DECIMAL(6,5),
    confidence_interval_95_upper DECIMAL(6,5),
    standard_error DECIMAL(6,5),
    
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure meaningful aggregations
    CONSTRAINT valid_sample_size CHECK (valid_responses > 0),
    UNIQUE (experiment_id, model_id, benchmark_id, scrambling_method, scrambling_intensity, domain, question_type)
);

-- Statistical analysis results
CREATE TABLE statistical_analyses (
    analysis_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    analysis_type VARCHAR(50) NOT NULL, -- 'anova', 't_test', 'correlation', 'regression'
    analysis_name VARCHAR(100) NOT NULL,
    
    -- Analysis scope
    models_included UUID[] NOT NULL, -- Array of model IDs
    factors JSONB NOT NULL, -- Independent variables tested
    dependent_variable VARCHAR(50) NOT NULL,
    
    -- Results
    test_statistic DECIMAL(10,6),
    p_value DECIMAL(15,10) CHECK (p_value BETWEEN 0.0 AND 1.0),
    effect_size DECIMAL(6,4),
    degrees_of_freedom INTEGER,
    
    -- Additional results (analysis-specific)
    detailed_results JSONB NOT NULL, -- F-ratios, post-hoc tests, etc.
    assumptions_tested JSONB, -- Normality, homogeneity tests
    interpretation TEXT,
    
    -- Quality indicators
    alpha_level DECIMAL(4,3) NOT NULL DEFAULT 0.05,
    power_analysis DECIMAL(4,3), -- Statistical power if computed
    
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    computed_by VARCHAR(100) NOT NULL,
    
    UNIQUE (experiment_id, analysis_type, analysis_name)
);

-- Threshold analysis results (key for language dependency research)
CREATE TABLE threshold_analyses (
    threshold_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    model_id UUID NOT NULL REFERENCES models(model_id),
    benchmark_id UUID REFERENCES benchmarks(benchmark_id),
    
    -- Threshold detection parameters
    scrambling_method VARCHAR(50) NOT NULL,
    performance_metric VARCHAR(50) NOT NULL DEFAULT 'accuracy',
    threshold_definition VARCHAR(100) NOT NULL, -- e.g., "50% of baseline performance"
    
    -- Detected thresholds
    threshold_value DECIMAL(5,2), -- Scrambling intensity where threshold is crossed
    threshold_confidence_interval_lower DECIMAL(5,2),
    threshold_confidence_interval_upper DECIMAL(5,2),
    
    -- Performance at threshold
    baseline_performance DECIMAL(6,5) NOT NULL, -- Performance at 0% scrambling
    threshold_performance DECIMAL(6,5), -- Performance at threshold
    performance_drop DECIMAL(6,5), -- Absolute drop in performance
    
    -- Statistical significance
    is_significant BOOLEAN NOT NULL,
    p_value DECIMAL(15,10),
    test_method VARCHAR(50), -- Method used to detect threshold
    
    -- Language dependency coefficient
    dependency_coefficient DECIMAL(6,5), -- How much performance depends on text structure
    coefficient_interpretation VARCHAR(20) CHECK (coefficient_interpretation IN ('low', 'medium', 'high', 'extreme')),
    
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    
    UNIQUE (experiment_id, model_id, benchmark_id, scrambling_method, performance_metric)
);

-- Domain-specific performance breakdown
CREATE TABLE domain_performance (
    domain_perf_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    model_id UUID NOT NULL REFERENCES models(model_id),
    
    domain VARCHAR(50) NOT NULL,
    subdomain VARCHAR(50),
    
    -- Performance across scrambling levels
    baseline_accuracy DECIMAL(6,5) NOT NULL, -- 0% scrambling
    low_scrambling_accuracy DECIMAL(6,5), -- 1-25% scrambling
    medium_scrambling_accuracy DECIMAL(6,5), -- 26-50% scrambling
    high_scrambling_accuracy DECIMAL(6,5), -- 51-75% scrambling
    extreme_scrambling_accuracy DECIMAL(6,5), -- 76-100% scrambling
    
    -- Robustness metrics
    robustness_score DECIMAL(6,5), -- Overall resistance to scrambling
    degradation_rate DECIMAL(8,5), -- Rate of performance loss per scrambling unit
    
    -- Sample sizes
    total_questions INTEGER NOT NULL CHECK (total_questions > 0),
    questions_per_level INTEGER NOT NULL,
    
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE (experiment_id, model_id, domain, subdomain)
);

-- ==============================================================================
-- RESEARCH-SPECIFIC TABLES
-- ==============================================================================

-- Cognitive load classifications (research-specific feature)
CREATE TABLE cognitive_load_classifications (
    classification_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    question_id UUID NOT NULL REFERENCES questions(question_id) ON DELETE CASCADE,
    
    -- Cognitive load factors
    working_memory_load VARCHAR(20) NOT NULL CHECK (working_memory_load IN ('low', 'medium', 'high')),
    processing_complexity VARCHAR(20) NOT NULL CHECK (processing_complexity IN ('low', 'medium', 'high')),
    attention_demand VARCHAR(20) NOT NULL CHECK (attention_demand IN ('low', 'medium', 'high')),
    
    -- Composite scores
    overall_cognitive_load DECIMAL(4,2) NOT NULL CHECK (overall_cognitive_load BETWEEN 1.0 AND 10.0),
    load_category VARCHAR(20) NOT NULL CHECK (load_category IN ('minimal', 'light', 'moderate', 'heavy', 'extreme')),
    
    -- Classification method
    classification_method VARCHAR(50) NOT NULL, -- 'expert_judgment', 'automated', 'crowd_sourced'
    classifier_id VARCHAR(100), -- Who/what did the classification
    confidence_level DECIMAL(4,3) CHECK (confidence_level BETWEEN 0.0 AND 1.0),
    
    -- Supporting evidence
    justification TEXT,
    classification_features JSONB, -- Specific features that drove classification
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE (question_id, classification_method, classifier_id)
);

-- Scrambling transformation details (for reproducibility)
CREATE TABLE scrambling_transformations (
    transformation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scrambled_question_id UUID NOT NULL REFERENCES scrambled_questions(scrambled_id) ON DELETE CASCADE,
    
    -- Transformation sequence
    step_number INTEGER NOT NULL CHECK (step_number > 0),
    transformation_type VARCHAR(50) NOT NULL, -- 'char_shuffle', 'word_shuffle', etc.
    
    -- Before/after for this step
    input_text TEXT NOT NULL,
    output_text TEXT NOT NULL,
    
    -- Step-specific parameters
    parameters JSONB NOT NULL,
    affected_positions INTEGER[], -- Which character/word positions were modified
    
    -- Validation
    reversible BOOLEAN NOT NULL DEFAULT false,
    validation_checksum VARCHAR(64), -- For integrity checking
    
    UNIQUE (scrambled_question_id, step_number)
);

-- Prompt templates for standardization
CREATE TABLE prompt_templates (
    template_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    template_name VARCHAR(100) NOT NULL UNIQUE,
    template_version VARCHAR(20) NOT NULL DEFAULT '1.0',
    
    -- Template content
    system_prompt TEXT,
    user_prompt_template TEXT NOT NULL, -- With placeholders like {question}
    
    -- Template metadata
    intended_use TEXT NOT NULL,
    model_compatibility VARCHAR(100)[], -- Which models this works well with
    
    -- Variables and formatting
    required_variables VARCHAR(50)[] NOT NULL, -- Variables that must be provided
    optional_variables VARCHAR(50)[],
    formatting_rules JSONB,
    
    -- Quality assurance
    is_validated BOOLEAN NOT NULL DEFAULT false,
    validation_results JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) NOT NULL,
    
    UNIQUE (template_name, template_version)
);

-- ==============================================================================
-- VISUALIZATION AND EXPORT SUPPORT
-- ==============================================================================

-- Pre-computed data for common visualizations
CREATE TABLE visualization_data (
    viz_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    
    -- Visualization type and parameters
    chart_type VARCHAR(50) NOT NULL, -- 'performance_curve', 'threshold_comparison', etc.
    chart_title VARCHAR(200) NOT NULL,
    dimensions JSONB NOT NULL, -- Grouping variables
    
    -- Data points
    data_points JSONB NOT NULL, -- Array of {x, y, metadata} objects
    aggregation_level VARCHAR(50) NOT NULL, -- 'individual', 'model_avg', 'family_avg'
    
    -- Styling and display hints
    suggested_colors VARCHAR(20)[],
    axis_labels JSONB, -- {x: "label", y: "label"}
    display_hints JSONB, -- Chart-specific rendering hints
    
    -- Export formats
    csv_data TEXT, -- Pre-formatted CSV for easy export
    r_code TEXT, -- R code to recreate the visualization
    python_code TEXT, -- Python/matplotlib code
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE (experiment_id, chart_type, chart_title)
);

-- ==============================================================================
-- INDEXES FOR PERFORMANCE OPTIMIZATION
-- ==============================================================================

-- Primary query patterns for research analysis

-- Experiment-based queries
CREATE INDEX idx_responses_experiment_model ON responses(experiment_id, model_id);
CREATE INDEX idx_responses_experiment_scrambling ON responses(experiment_id, scrambled_question_id) 
    WHERE scrambled_question_id IS NOT NULL;

-- Model performance queries
CREATE INDEX idx_performance_metrics_model_scrambling ON performance_metrics(model_id, scrambling_method, scrambling_intensity);
CREATE INDEX idx_responses_model_accuracy ON responses(model_id, is_correct) WHERE is_correct IS NOT NULL;

-- Statistical analysis queries
CREATE INDEX idx_responses_timing ON responses(response_time_ms) WHERE response_time_ms IS NOT NULL;
CREATE INDEX idx_questions_domain ON questions(domain, subdomain);
CREATE INDEX idx_scrambled_intensity ON scrambled_questions(scrambling_intensity, scrambling_method);

-- Research-specific queries
CREATE INDEX idx_cognitive_load ON cognitive_load_classifications(overall_cognitive_load, load_category);
CREATE INDEX idx_threshold_analysis_model ON threshold_analyses(model_id, scrambling_method, is_significant);

-- Composite indexes for complex queries
CREATE INDEX idx_responses_complete_analysis ON responses(experiment_id, model_id, is_correct, response_time_ms) 
    WHERE is_correct IS NOT NULL;
CREATE INDEX idx_performance_metrics_analysis ON performance_metrics(experiment_id, scrambling_intensity, accuracy);

-- Full-text search for research queries
CREATE INDEX idx_experiments_search ON experiments USING gin(to_tsvector('english', experiment_name || ' ' || description));
CREATE INDEX idx_questions_search ON questions USING gin(to_tsvector('english', question_text));

-- ==============================================================================
-- VIEWS FOR COMMON RESEARCH OPERATIONS
-- ==============================================================================

-- Language dependency coefficients by model
CREATE VIEW language_dependency_summary AS
SELECT 
    e.experiment_name,
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
JOIN experiments e ON ta.experiment_id = e.experiment_id
JOIN models m ON ta.model_id = m.model_id
WHERE ta.is_significant = true
ORDER BY m.model_family, ta.dependency_coefficient DESC;

-- Performance curves for visualization
CREATE VIEW performance_curves AS
SELECT 
    e.experiment_name,
    m.model_family,
    m.model_name,
    pm.scrambling_method,
    pm.scrambling_intensity,
    pm.accuracy,
    pm.confidence_interval_95_lower,
    pm.confidence_interval_95_upper,
    pm.total_responses
FROM performance_metrics pm
JOIN experiments e ON pm.experiment_id = e.experiment_id
JOIN models m ON pm.model_id = m.model_id
ORDER BY m.model_family, m.model_name, pm.scrambling_intensity;

-- Model comparison summary
CREATE VIEW model_comparison_summary AS
SELECT 
    e.experiment_name,
    m.model_family,
    m.model_name,
    m.parameter_count,
    COUNT(DISTINCT r.response_id) as total_responses,
    AVG(CASE WHEN r.is_correct THEN 1.0 ELSE 0.0 END) as overall_accuracy,
    AVG(r.response_time_ms) as avg_response_time,
    SUM(COALESCE(r.cost, 0)) as total_cost,
    MAX(ta.dependency_coefficient) as max_dependency_coefficient
FROM experiments e
JOIN responses r ON e.experiment_id = r.experiment_id
JOIN models m ON r.model_id = m.model_id
LEFT JOIN threshold_analyses ta ON e.experiment_id = ta.experiment_id AND m.model_id = ta.model_id
WHERE r.is_correct IS NOT NULL
GROUP BY e.experiment_id, e.experiment_name, m.model_id, m.model_family, m.model_name, m.parameter_count
ORDER BY overall_accuracy DESC;

-- Research reproducibility view
CREATE VIEW experiment_reproducibility AS
SELECT 
    e.experiment_id,
    e.experiment_name,
    e.git_commit_hash,
    e.git_branch,
    e.environment_snapshot->>'python_version' as python_version,
    e.random_seed,
    COUNT(DISTINCT r.model_id) as models_tested,
    COUNT(DISTINCT r.response_id) as total_responses,
    e.created_at,
    e.completed_at,
    e.status
FROM experiments e
LEFT JOIN responses r ON e.experiment_id = r.experiment_id
GROUP BY e.experiment_id, e.experiment_name, e.git_commit_hash, e.git_branch, 
         e.environment_snapshot, e.random_seed, e.created_at, e.completed_at, e.status
ORDER BY e.created_at DESC;

-- ==============================================================================
-- FUNCTIONS FOR COMMON CALCULATIONS
-- ==============================================================================

-- Calculate language dependency coefficient
CREATE OR REPLACE FUNCTION calculate_language_dependency_coefficient(
    p_experiment_id UUID,
    p_model_id UUID,
    p_scrambling_method VARCHAR(50) DEFAULT 'character'
) RETURNS DECIMAL(6,5) AS $$
DECLARE
    baseline_acc DECIMAL(6,5);
    avg_scrambled_acc DECIMAL(6,5);
    coefficient DECIMAL(6,5);
BEGIN
    -- Get baseline accuracy (0% scrambling or original questions)
    SELECT AVG(CASE WHEN r.is_correct THEN 1.0 ELSE 0.0 END)
    INTO baseline_acc
    FROM responses r
    LEFT JOIN scrambled_questions sq ON r.scrambled_question_id = sq.scrambled_id
    WHERE r.experiment_id = p_experiment_id
      AND r.model_id = p_model_id
      AND r.is_correct IS NOT NULL
      AND (r.original_question_id IS NOT NULL OR 
           (sq.scrambled_id IS NOT NULL AND sq.scrambling_intensity = 0.0));
    
    -- Get average accuracy across all scrambling levels > 0
    SELECT AVG(CASE WHEN r.is_correct THEN 1.0 ELSE 0.0 END)
    INTO avg_scrambled_acc
    FROM responses r
    JOIN scrambled_questions sq ON r.scrambled_question_id = sq.scrambled_id
    WHERE r.experiment_id = p_experiment_id
      AND r.model_id = p_model_id
      AND r.is_correct IS NOT NULL
      AND sq.scrambling_method = p_scrambling_method
      AND sq.scrambling_intensity > 0.0;
    
    -- Calculate coefficient (1 - performance_retention)
    IF baseline_acc > 0 THEN
        coefficient := 1.0 - (avg_scrambled_acc / baseline_acc);
    ELSE
        coefficient := NULL;
    END IF;
    
    RETURN GREATEST(0.0, LEAST(1.0, coefficient));
END;
$$ LANGUAGE plpgsql;

-- ==============================================================================
-- DATA INTEGRITY CONSTRAINTS AND TRIGGERS
-- ==============================================================================

-- Ensure response timing is reasonable
CREATE OR REPLACE FUNCTION validate_response_timing() RETURNS trigger AS $$
BEGIN
    -- Response time should be reasonable (between 1ms and 10 minutes)
    IF NEW.response_time_ms < 1 OR NEW.response_time_ms > 600000 THEN
        RAISE EXCEPTION 'Response time % is outside reasonable bounds (1ms - 10min)', NEW.response_time_ms;
    END IF;
    
    -- Token count should match response length approximately
    IF NEW.token_count IS NOT NULL AND LENGTH(NEW.raw_response) > 0 THEN
        IF NEW.token_count > LENGTH(NEW.raw_response) * 2 OR NEW.token_count < LENGTH(NEW.raw_response) / 10 THEN
            RAISE WARNING 'Token count % seems inconsistent with response length %', 
                         NEW.token_count, LENGTH(NEW.raw_response);
        END IF;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER validate_response_timing_trigger
    BEFORE INSERT OR UPDATE ON responses
    FOR EACH ROW EXECUTE FUNCTION validate_response_timing();

-- Auto-update performance metrics when responses change
CREATE OR REPLACE FUNCTION refresh_performance_metrics() RETURNS trigger AS $$
BEGIN
    -- Delete existing metrics for this combination
    DELETE FROM performance_metrics 
    WHERE experiment_id = COALESCE(NEW.experiment_id, OLD.experiment_id)
      AND model_id = COALESCE(NEW.model_id, OLD.model_id);
    
    -- Note: In production, you'd want to recalculate specific metrics
    -- This is a placeholder for a more sophisticated update mechanism
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Note: This trigger would be expensive on large datasets
-- In production, consider batch updates or async processing
-- CREATE TRIGGER refresh_performance_metrics_trigger
--     AFTER INSERT OR UPDATE OR DELETE ON responses
--     FOR EACH ROW EXECUTE FUNCTION refresh_performance_metrics();

-- ==============================================================================
-- SAMPLE DATA INSERTION FUNCTIONS
-- ==============================================================================

-- Helper function to create a sample experiment
CREATE OR REPLACE FUNCTION create_sample_experiment(
    p_name VARCHAR(255),
    p_description TEXT,
    p_researcher VARCHAR(100) DEFAULT 'sample_researcher'
) RETURNS UUID AS $$
DECLARE
    experiment_uuid UUID;
BEGIN
    INSERT INTO experiments (
        experiment_name,
        description,
        research_question,
        git_commit_hash,
        git_branch,
        environment_snapshot,
        random_seed,
        researcher_name,
        configuration
    ) VALUES (
        p_name,
        p_description,
        'How does text scrambling affect model performance?',
        '1234567890abcdef1234567890abcdef12345678',
        'main',
        '{"python_version": "3.11.0", "numpy_version": "1.24.0", "pandas_version": "2.0.0"}',
        42,
        p_researcher,
        '{"max_scrambling_intensity": 100, "scrambling_methods": ["character", "word"], "models_to_test": ["gpt-3.5", "llama-2"]}'
    ) RETURNING experiment_id INTO experiment_uuid;
    
    RETURN experiment_uuid;
END;
$$ LANGUAGE plpgsql;

-- ==============================================================================
-- EXPORT AND REPORTING FUNCTIONS
-- ==============================================================================

-- Generate CSV export for statistical analysis
CREATE OR REPLACE FUNCTION export_experiment_data(
    p_experiment_id UUID,
    p_include_raw_responses BOOLEAN DEFAULT false
) RETURNS TEXT AS $$
DECLARE
    csv_data TEXT;
BEGIN
    -- Generate CSV header and data
    WITH export_data AS (
        SELECT 
            e.experiment_name,
            m.model_family,
            m.model_name,
            COALESCE(sq.scrambling_method, 'original') as scrambling_method,
            COALESCE(sq.scrambling_intensity, 0.0) as scrambling_intensity,
            q.domain,
            q.question_type,
            CASE WHEN r.is_correct THEN 1 ELSE 0 END as is_correct,
            r.partial_credit,
            r.response_time_ms,
            CASE WHEN p_include_raw_responses THEN r.raw_response ELSE '[EXCLUDED]' END as raw_response
        FROM responses r
        JOIN experiments e ON r.experiment_id = e.experiment_id
        JOIN models m ON r.model_id = m.model_id
        LEFT JOIN questions q ON r.original_question_id = q.question_id
        LEFT JOIN scrambled_questions sq ON r.scrambled_question_id = sq.scrambled_id
        LEFT JOIN questions q2 ON sq.original_question_id = q2.question_id
        WHERE r.experiment_id = p_experiment_id
          AND r.is_correct IS NOT NULL
    )
    SELECT string_agg(
        experiment_name || ',' || 
        model_family || ',' || 
        model_name || ',' || 
        scrambling_method || ',' || 
        scrambling_intensity::text || ',' ||
        COALESCE(domain, '') || ',' ||
        COALESCE(question_type, '') || ',' ||
        is_correct::text || ',' ||
        COALESCE(partial_credit::text, '') || ',' ||
        response_time_ms::text || ',' ||
        '"' || replace(raw_response, '"', '""') || '"',
        E'\n'
    ) INTO csv_data
    FROM export_data;
    
    -- Add header
    csv_data := 'experiment_name,model_family,model_name,scrambling_method,scrambling_intensity,domain,question_type,is_correct,partial_credit,response_time_ms,raw_response' || E'\n' || csv_data;
    
    RETURN csv_data;
END;
$$ LANGUAGE plpgsql;

-- ==============================================================================
-- MAINTENANCE AND MONITORING
-- ==============================================================================

-- Database statistics and health check
CREATE OR REPLACE VIEW database_health AS
SELECT 
    'experiments' as table_name,
    COUNT(*) as row_count,
    COUNT(*) FILTER (WHERE status = 'completed') as completed_experiments,
    MAX(created_at) as latest_entry
FROM experiments
UNION ALL
SELECT 
    'responses' as table_name,
    COUNT(*) as row_count,
    COUNT(*) FILTER (WHERE is_correct IS NOT NULL) as evaluated_responses,
    MAX(created_at) as latest_entry
FROM responses
UNION ALL
SELECT 
    'performance_metrics' as table_name,
    COUNT(*) as row_count,
    COUNT(DISTINCT experiment_id) as experiments_analyzed,
    MAX(computed_at) as latest_entry
FROM performance_metrics
UNION ALL
SELECT 
    'threshold_analyses' as table_name,
    COUNT(*) as row_count,
    COUNT(*) FILTER (WHERE is_significant = true) as significant_thresholds,
    MAX(computed_at) as latest_entry
FROM threshold_analyses;

-- ==============================================================================
-- EXAMPLE QUERIES FOR COMMON RESEARCH OPERATIONS
-- ==============================================================================

/*
-- Example 1: Calculate language dependency coefficients across model families
SELECT 
    m.model_family,
    AVG(calculate_language_dependency_coefficient(r.experiment_id, r.model_id, 'character')) as avg_dependency_coeff,
    COUNT(DISTINCT r.model_id) as models_tested
FROM responses r
JOIN models m ON r.model_id = m.model_id
WHERE r.experiment_id = '550e8400-e29b-41d4-a716-446655440000' -- Replace with actual experiment ID
GROUP BY m.model_family
ORDER BY avg_dependency_coeff DESC;

-- Example 2: Generate performance curve data for visualization
SELECT 
    model_name,
    scrambling_intensity,
    accuracy,
    confidence_interval_95_lower,
    confidence_interval_95_upper
FROM performance_curves
WHERE experiment_name = 'Language Dependency Baseline Study'
  AND scrambling_method = 'character'
ORDER BY model_name, scrambling_intensity;

-- Example 3: Compare threshold values between model families
SELECT 
    model_family,
    scrambling_method,
    AVG(threshold_value) as avg_threshold,
    STDDEV(threshold_value) as threshold_std,
    COUNT(*) as models_tested
FROM language_dependency_summary
GROUP BY model_family, scrambling_method
HAVING COUNT(*) >= 3  -- Only families with multiple models
ORDER BY avg_threshold DESC;

-- Example 4: Export experiment data for external analysis
SELECT export_experiment_data('550e8400-e29b-41d4-a716-446655440000', false);

-- Example 5: Find experiments needing replication
SELECT 
    experiment_name,
    COUNT(DISTINCT model_id) as models_tested,
    MAX(completed_at) as completion_date,
    git_commit_hash
FROM experiment_reproducibility
WHERE status = 'completed'
  AND models_tested < 5  -- Might need more models
ORDER BY completion_date DESC;

-- Example 6: Performance analysis by cognitive load
SELECT 
    cl.load_category,
    pm.scrambling_intensity,
    AVG(pm.accuracy) as avg_accuracy,
    COUNT(*) as sample_size
FROM performance_metrics pm
JOIN responses r ON pm.experiment_id = r.experiment_id AND pm.model_id = r.model_id
JOIN questions q ON r.original_question_id = q.question_id
JOIN cognitive_load_classifications cl ON q.question_id = cl.question_id
WHERE pm.scrambling_method = 'character'
GROUP BY cl.load_category, pm.scrambling_intensity
HAVING COUNT(*) >= 10  -- Minimum sample size
ORDER BY cl.load_category, pm.scrambling_intensity;
*/

-- ==============================================================================
-- FINAL NOTES AND DOCUMENTATION
-- ==============================================================================

COMMENT ON DATABASE language_dependency_atlas IS 'Academic research database for Language Dependency Atlas project studying the effects of text scrambling on AI model performance';

COMMENT ON TABLE experiments IS 'Top-level container for research experiments with full reproducibility metadata';
COMMENT ON TABLE models IS 'Language models being tested with technical specifications and access details';
COMMENT ON TABLE responses IS 'Individual model responses to questions with performance metrics and evaluation results';
COMMENT ON TABLE performance_metrics IS 'Pre-computed aggregated performance data for efficient statistical analysis';
COMMENT ON TABLE threshold_analyses IS 'Language dependency threshold detection results for each model and scrambling method';
COMMENT ON TABLE statistical_analyses IS 'Results from statistical tests (ANOVA, t-tests, correlations) for academic publication';

COMMENT ON FUNCTION calculate_language_dependency_coefficient IS 'Calculates how much a models performance depends on text structure (0=no dependency, 1=complete dependency)';
COMMENT ON VIEW language_dependency_summary IS 'Summary of language dependency coefficients by model for quick analysis';
COMMENT ON VIEW performance_curves IS 'Data formatted for plotting performance vs scrambling intensity curves';

-- Final success message
SELECT 'Language Dependency Atlas database schema created successfully!' as status,
       'Schema includes ' || COUNT(*) || ' tables with comprehensive research support' as details
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name NOT LIKE 'pg_%' 
  AND table_name NOT LIKE 'sql_%';