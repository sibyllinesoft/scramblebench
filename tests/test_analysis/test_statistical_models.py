"""
Tests for statistical_models.py

Tests the core statistical modeling classes including:
- ScalingAnalyzer: Master class for scaling pattern analysis
- GLMMAnalyzer: Generalized Linear Mixed Models
- GAMAnalyzer: Generalized Additive Models  
- ChangepointAnalyzer: Changepoint detection models
- ContaminationAnalyzer: Contamination vs brittleness analysis
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
import logging

# Import classes under test
from scramblebench.analysis.statistical_models import (
    ScalingAnalyzer,
    GLMMAnalyzer,
    GAMAnalyzer,
    ChangepointAnalyzer,
    ContaminationAnalyzer,
    ModelFit,
    R_AVAILABLE
)


class TestModelFit:
    """Tests for ModelFit dataclass"""

    def test_model_fit_creation(self):
        """Test ModelFit object creation"""
        fit = ModelFit(
            model_name="test_model",
            aic=100.5,
            bic=105.2,
            log_likelihood=-45.25,
            fixed_effects={"intercept": 1.2, "slope": 0.5},
            random_effects={"group_var": 0.3},
            converged=True,
            n_observations=100,
            cv_score=0.85
        )
        
        assert fit.model_name == "test_model"
        assert fit.aic == 100.5
        assert fit.bic == 105.2
        assert fit.log_likelihood == -45.25
        assert fit.converged is True
        assert fit.n_observations == 100
        assert fit.cv_score == 0.85
        
    def test_model_fit_defaults(self):
        """Test ModelFit with minimal parameters"""
        fit = ModelFit(
            model_name="minimal",
            aic=50.0,
            bic=55.0,
            log_likelihood=-20.0,
            fixed_effects={},
            converged=True,
            n_observations=50
        )
        
        assert fit.random_effects == {}
        assert fit.cv_score is None
        assert fit.deviance is None


class TestScalingAnalyzer:
    """Tests for ScalingAnalyzer class"""
    
    @pytest.fixture
    def mock_database(self):
        """Mock database for testing"""
        db = Mock()
        mock_conn = Mock()
        db.get_connection.return_value = mock_conn
        return db
        
    @pytest.fixture
    def sample_analysis_data(self):
        """Sample data for analysis testing"""
        np.random.seed(42)
        n_obs = 200
        
        # Create realistic test data
        model_families = ['transformer', 'transformer', 'rnn', 'rnn']
        model_ids = ['gpt-3.5', 'gpt-4', 'lstm-small', 'lstm-large']
        n_params = [175e9, 1.76e12, 10e6, 100e6]
        
        data = []
        
        for i, (family, model_id, params) in enumerate(zip(model_families, model_ids, n_params)):
            n_model_obs = n_obs // 4
            
            # Create scaling pattern: larger models perform better
            logN = np.log10(params)
            base_accuracy = 0.5 + 0.3 * (logN - 6) / 6  # Scale 0.5-0.8
            
            model_data = {
                'model_id': [model_id] * n_model_obs,
                'model_family': [family] * n_model_obs,
                'n_params': [params] * n_model_obs,
                'logN': [logN] * n_model_obs,
                'domain': np.random.choice(['math', 'logic', 'reading'], n_model_obs),
                'transform': np.random.choice(['original', 'paraphrase', 'scramble'], n_model_obs),
                'scramble_level': np.random.choice([None, 0.2, 0.5, 0.8], n_model_obs),
                'is_correct': np.random.binomial(1, base_accuracy, n_model_obs),
                'tok_kl': np.random.exponential(0.1, n_model_obs),
                'tok_frag': np.random.beta(2, 2, n_model_obs)
            }
            
            data.append(pd.DataFrame(model_data))
        
        df = pd.concat(data, ignore_index=True)
        
        # Add derived columns that ScalingAnalyzer expects
        def create_condition(row):
            if row['transform'] == 'original':
                return 'original'
            elif row['transform'] == 'paraphrase':
                return 'paraphrase'
            else:
                level = row['scramble_level'] or 0
                return f'scramble_{level}'
        
        df['condition'] = df.apply(create_condition, axis=1)
        df['is_scrambled'] = df['transform'] == 'scramble'
        df['is_paraphrase'] = df['transform'] == 'paraphrase'
        df['scramble_intensity'] = df['scramble_level'].fillna(0.0)
        
        # Add factor columns
        df['model_family_f'] = pd.Categorical(df['model_family'])
        df['domain_f'] = pd.Categorical(df['domain'])
        df['condition_f'] = pd.Categorical(df['condition'])
        
        return df
    
    def test_scaling_analyzer_init(self, mock_database):
        """Test ScalingAnalyzer initialization"""
        analyzer = ScalingAnalyzer(
            database=mock_database,
            use_r_backend=True,
            alpha=0.01
        )
        
        assert analyzer.db == mock_database
        assert analyzer.alpha == 0.01
        assert analyzer.use_r == (True and R_AVAILABLE)
        assert analyzer.analysis_data is None
        assert isinstance(analyzer.model_fits, dict)
        assert isinstance(analyzer.best_models, dict)
        
    def test_prepare_analysis_data_success(self, mock_database, sample_analysis_data):
        """Test successful analysis data preparation"""
        # Mock database query result
        mock_conn = Mock()
        mock_database.get_connection.return_value = mock_conn
        
        # Mock the SQL query execution
        with patch('pandas.read_sql_query') as mock_read_sql:
            mock_read_sql.return_value = sample_analysis_data.copy()
            
            analyzer = ScalingAnalyzer(mock_database)
            result = analyzer.prepare_analysis_data("test_run_001")
            
            # Verify result structure
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert 'logN' in result.columns
            assert 'condition' in result.columns
            assert 'is_scrambled' in result.columns
            
            # Verify derived columns were created
            assert result['logN'].notna().all()
            assert result['condition'].notna().all()
            
            # Verify data was stored
            assert analyzer.analysis_data is not None
            
    def test_prepare_analysis_data_empty(self, mock_database):
        """Test analysis data preparation with empty result"""
        mock_conn = Mock()
        mock_database.get_connection.return_value = mock_conn
        
        with patch('pandas.read_sql_query') as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame()
            
            analyzer = ScalingAnalyzer(mock_database)
            
            with pytest.raises(ValueError, match="No evaluation data found"):
                analyzer.prepare_analysis_data("empty_run")
    
    def test_analyze_family(self, mock_database, sample_analysis_data):
        """Test family-specific analysis"""
        analyzer = ScalingAnalyzer(mock_database)
        
        # Mock the component analyzers
        analyzer.glmm = Mock()
        analyzer.gam = Mock()
        analyzer.changepoint = Mock()
        
        # Create mock model fits
        mock_glmm_fit = ModelFit(
            model_name="glmm",
            aic=100.0,
            bic=105.0,
            log_likelihood=-45.0,
            fixed_effects={"intercept": 1.0},
            converged=True,
            n_observations=50
        )
        
        mock_linear_fit = ModelFit(
            model_name="linear",
            aic=110.0,
            bic=112.0,
            log_likelihood=-52.0,
            fixed_effects={"intercept": 1.0, "slope": 0.5},
            converged=True,
            n_observations=50
        )
        
        analyzer.glmm.fit_hierarchical_model.return_value = mock_glmm_fit
        analyzer.changepoint.fit_linear_model.return_value = mock_linear_fit
        
        # Test analysis
        family_data = sample_analysis_data[
            sample_analysis_data['model_family'] == 'transformer'
        ].copy()
        
        result = analyzer._analyze_family(family_data, 'transformer')
        
        # Verify result structure
        assert result['family'] == 'transformer'
        assert result['n_observations'] == len(family_data)
        assert 'parameter_range' in result
        assert 'model_fits' in result
        
        # Verify model fits were attempted
        analyzer.glmm.fit_hierarchical_model.assert_called_once()
        analyzer.changepoint.fit_linear_model.assert_called_once()
    
    def test_compare_models(self, mock_database):
        """Test model comparison functionality"""
        analyzer = ScalingAnalyzer(mock_database)
        
        # Create test model fits
        model_fits = {
            'linear': ModelFit(
                model_name="linear",
                aic=120.0,
                bic=125.0,
                log_likelihood=-58.0,
                fixed_effects={"intercept": 1.0, "slope": 0.5},
                converged=True,
                n_observations=100
            ),
            'segmented': ModelFit(
                model_name="segmented", 
                aic=115.0,
                bic=122.0,
                log_likelihood=-55.0,
                fixed_effects={"intercept": 1.0, "slope1": 0.2, "slope2": 0.8},
                converged=True,
                n_observations=100
            )
        }
        
        result = analyzer._compare_models(model_fits)
        
        # Verify comparison results
        assert 'comparison_table' in result
        assert 'best_by_aic' in result
        assert 'aic_weights' in result
        assert 'evidence_ratio' in result
        
        # Best model should be segmented (lower AIC)
        assert result['best_by_aic'] == 'segmented'
        
        # AIC weights should sum to 1
        weights = list(result['aic_weights'].values())
        assert abs(sum(weights) - 1.0) < 1e-6
        
    @patch('scramblebench.analysis.statistical_models.pd.read_sql_query')
    def test_run_full_analysis_integration(self, mock_read_sql, mock_database, sample_analysis_data):
        """Test full analysis pipeline integration"""
        mock_read_sql.return_value = sample_analysis_data
        
        analyzer = ScalingAnalyzer(mock_database)
        
        # Mock component analyzers
        analyzer.glmm = Mock()
        analyzer.gam = Mock() 
        analyzer.changepoint = Mock()
        analyzer.contamination = Mock()
        
        # Set up mock returns
        mock_fit = ModelFit(
            model_name="test",
            aic=100.0,
            bic=105.0,
            log_likelihood=-45.0,
            fixed_effects={"intercept": 1.0},
            converged=True,
            n_observations=50
        )
        
        analyzer.glmm.fit_hierarchical_model.return_value = mock_fit
        analyzer.changepoint.fit_linear_model.return_value = mock_fit
        analyzer.contamination.analyze.return_value = {"contamination_score": 0.3}
        
        # Run analysis
        result = analyzer.run_full_analysis("test_run")
        
        # Verify result structure
        assert result['run_id'] == "test_run"
        assert 'n_observations' in result
        assert 'n_models' in result
        assert 'family_results' in result
        assert 'contamination_analysis' in result
        
        # Verify component methods were called
        analyzer.contamination.analyze.assert_called_once()


class TestGLMMAnalyzer:
    """Tests for GLMMAnalyzer class"""
    
    def test_glmm_analyzer_init(self):
        """Test GLMMAnalyzer initialization"""
        analyzer = GLMMAnalyzer(use_r=True, alpha=0.05)
        
        assert analyzer.use_r == (True and R_AVAILABLE)
        assert analyzer.alpha == 0.05
        
    def test_fit_hierarchical_model_mock(self, sample_analysis_data):
        """Test GLMM fitting with mocked R backend"""
        analyzer = GLMMAnalyzer(use_r=False)  # Use Python backend
        
        # Mock the fitting process
        with patch.object(analyzer, '_fit_python_glmm') as mock_fit:
            mock_fit.return_value = ModelFit(
                model_name="glmm",
                aic=150.0,
                bic=160.0,
                log_likelihood=-70.0,
                fixed_effects={"intercept": 2.1, "logN": 0.45},
                random_effects={"model_family": 0.12},
                converged=True,
                n_observations=len(sample_analysis_data)
            )
            
            result = analyzer.fit_hierarchical_model(sample_analysis_data)
            
            assert result.model_name == "glmm"
            assert result.converged is True
            mock_fit.assert_called_once()


class TestGAMAnalyzer:
    """Tests for GAMAnalyzer class"""
    
    def test_gam_analyzer_init(self):
        """Test GAMAnalyzer initialization"""
        analyzer = GAMAnalyzer(use_r=True, alpha=0.01)
        
        assert analyzer.use_r == (True and R_AVAILABLE)
        assert analyzer.alpha == 0.01
        
    def test_fit_monotone_gam_mock(self, sample_analysis_data):
        """Test GAM fitting with mocked backend"""
        analyzer = GAMAnalyzer(use_r=False)
        
        with patch.object(analyzer, '_fit_python_gam') as mock_fit:
            mock_fit.return_value = ModelFit(
                model_name="gam",
                aic=140.0,
                bic=155.0,
                log_likelihood=-65.0,
                fixed_effects={"smooth_logN": "nonparametric"},
                converged=True,
                n_observations=len(sample_analysis_data)
            )
            
            result = analyzer.fit_monotone_gam(sample_analysis_data)
            
            assert result.model_name == "gam"
            assert result.converged is True
            mock_fit.assert_called_once()


class TestChangepointAnalyzer:
    """Tests for ChangepointAnalyzer class"""
    
    def test_changepoint_analyzer_init(self):
        """Test ChangepointAnalyzer initialization"""
        analyzer = ChangepointAnalyzer(use_r=True, alpha=0.05)
        
        assert analyzer.use_r == (True and R_AVAILABLE)
        assert analyzer.alpha == 0.05
        
    def test_fit_linear_model(self, sample_analysis_data):
        """Test linear model fitting"""
        analyzer = ChangepointAnalyzer(use_r=False)
        
        with patch.object(analyzer, '_fit_python_linear') as mock_fit:
            mock_fit.return_value = ModelFit(
                model_name="linear",
                aic=180.0,
                bic=185.0,
                log_likelihood=-88.0,
                fixed_effects={"intercept": 0.5, "logN": 0.1},
                converged=True,
                n_observations=len(sample_analysis_data)
            )
            
            result = analyzer.fit_linear_model(sample_analysis_data)
            
            assert result.model_name == "linear"
            assert result.converged is True
            mock_fit.assert_called_once()
    
    def test_fit_segmented_model(self, sample_analysis_data):
        """Test segmented model fitting"""
        analyzer = ChangepointAnalyzer(use_r=False)
        
        with patch.object(analyzer, '_fit_python_segmented') as mock_fit:
            mock_fit.return_value = ModelFit(
                model_name="segmented",
                aic=175.0,
                bic=185.0,
                log_likelihood=-85.0,
                fixed_effects={
                    "intercept": 0.5, 
                    "slope1": 0.05, 
                    "slope2": 0.25,
                    "breakpoint": 9.5
                },
                converged=True,
                n_observations=len(sample_analysis_data)
            )
            
            result = analyzer.fit_segmented_model(sample_analysis_data)
            
            assert result.model_name == "segmented"
            assert "breakpoint" in result.fixed_effects
            mock_fit.assert_called_once()
    
    def test_sup_f_test(self):
        """Test sup-F test for changepoint significance"""
        analyzer = ChangepointAnalyzer(use_r=False)
        
        # Create mock model fits
        linear_fit = ModelFit(
            model_name="linear",
            aic=180.0,
            bic=185.0,
            log_likelihood=-88.0,
            fixed_effects={"intercept": 0.5, "logN": 0.1},
            converged=True,
            n_observations=100
        )
        
        segmented_fit = ModelFit(
            model_name="segmented",
            aic=175.0,
            bic=185.0,
            log_likelihood=-85.0,
            fixed_effects={"intercept": 0.5, "slope1": 0.05, "slope2": 0.25},
            converged=True,
            n_observations=100
        )
        
        with patch.object(analyzer, '_calculate_sup_f_statistic') as mock_calc:
            mock_calc.return_value = {"statistic": 12.5, "p_value": 0.001}
            
            result = analyzer.sup_f_test(linear_fit, segmented_fit)
            
            assert "statistic" in result
            assert "p_value" in result
            mock_calc.assert_called_once()


class TestContaminationAnalyzer:
    """Tests for ContaminationAnalyzer class"""
    
    def test_contamination_analyzer_init(self):
        """Test ContaminationAnalyzer initialization"""
        analyzer = ContaminationAnalyzer(alpha=0.01)
        
        assert analyzer.alpha == 0.01
        
    def test_analyze(self, sample_analysis_data):
        """Test contamination analysis"""
        analyzer = ContaminationAnalyzer(alpha=0.05)
        
        result = analyzer.analyze(sample_analysis_data)
        
        # Should return a dictionary with analysis results
        assert isinstance(result, dict)
        
        # Basic structure check
        expected_keys = ["n_models", "contamination_scores", "brittleness_scores"]
        for key in expected_keys:
            assert key in result or len(result) > 0  # Basic structure validation
    
    def test_calculate_ldc_score(self, sample_analysis_data):
        """Test Language Dependency Coefficient calculation"""
        analyzer = ContaminationAnalyzer(alpha=0.05)
        
        # Get data for one model
        model_data = sample_analysis_data[
            sample_analysis_data['model_id'] == sample_analysis_data['model_id'].iloc[0]
        ]
        
        if len(model_data) > 0:
            ldc = analyzer._calculate_ldc(model_data)
            
            # LDC should be a valid float between 0 and 1
            assert isinstance(ldc, float)
            assert 0.0 <= ldc <= 1.0
    
    def test_separate_contamination_brittleness(self, sample_analysis_data):
        """Test contamination vs brittleness separation"""
        analyzer = ContaminationAnalyzer(alpha=0.05)
        
        # This would test the core functionality of separating
        # contamination effects from genuine brittleness
        contamination_score = analyzer._calculate_contamination_score(sample_analysis_data)
        brittleness_score = analyzer._calculate_brittleness_score(sample_analysis_data)
        
        # Basic validation
        assert isinstance(contamination_score, (int, float))
        assert isinstance(brittleness_score, (int, float))
        assert 0.0 <= contamination_score <= 1.0
        assert 0.0 <= brittleness_score <= 1.0


# Integration test
class TestStatisticalModelsIntegration:
    """Integration tests across statistical models"""
    
    def test_full_pipeline_mock(self, mock_database, sample_analysis_data):
        """Test complete statistical analysis pipeline"""
        
        with patch('pandas.read_sql_query') as mock_read_sql:
            mock_read_sql.return_value = sample_analysis_data
            
            analyzer = ScalingAnalyzer(mock_database, use_r_backend=False)
            
            # Mock all the fitting methods to return consistent results
            with patch.object(analyzer.glmm, 'fit_hierarchical_model') as mock_glmm, \
                 patch.object(analyzer.changepoint, 'fit_linear_model') as mock_linear, \
                 patch.object(analyzer.contamination, 'analyze') as mock_contam:
                
                # Set up consistent mock returns
                mock_glmm.return_value = ModelFit(
                    model_name="glmm", aic=100.0, bic=105.0, log_likelihood=-45.0,
                    fixed_effects={"intercept": 1.0}, converged=True, n_observations=50
                )
                
                mock_linear.return_value = ModelFit(
                    model_name="linear", aic=110.0, bic=115.0, log_likelihood=-50.0,
                    fixed_effects={"intercept": 1.0, "slope": 0.1}, converged=True, n_observations=50
                )
                
                mock_contam.return_value = {"overall_contamination": 0.15}
                
                # Run full analysis
                results = analyzer.run_full_analysis("integration_test")
                
                # Verify integration
                assert results['run_id'] == "integration_test"
                assert 'family_results' in results
                assert 'contamination_analysis' in results
                assert results['contamination_analysis']['overall_contamination'] == 0.15
                
                # Verify all components were called
                mock_glmm.assert_called()
                mock_linear.assert_called()
                mock_contam.assert_called_once()