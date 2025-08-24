"""
Comprehensive smoke tests for the paraphrase pipeline implementation.

Tests validate all components of Step S1 implementation:
- Provider isolation enforcement
- Database-integrated caching
- Quality control with semantic similarity ≥0.85 and surface divergence ≥0.25
- Async paraphrase generation with n=2 candidates and temp=0.3
- ≥95% acceptance rate target validation
- Academic-grade error handling and logging
"""

import asyncio
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from scramblebench.core.database import Database
from scramblebench.transforms.paraphrase_pipeline import (
    ParaphrasePipeline, 
    ProviderIsolationValidator,
    create_paraphrase_pipeline
)
from scramblebench.transforms.paraphrase_quality_reporter import ParaphraseQualityReporter
from scramblebench.core.exceptions import ValidationError


class TestProviderIsolationValidator:
    """Test provider isolation validation."""
    
    def test_provider_registration_success(self):
        """Test successful provider registration without conflicts."""
        validator = ProviderIsolationValidator()
        
        # Register different providers
        validator.register_paraphrase_provider("hosted_heldout")
        validator.register_evaluation_provider("ollama_local")
        
        assert "hosted_heldout" in validator.paraphrase_providers
        assert "ollama_local" in validator.evaluation_providers
        
        # Check isolation report
        report = validator.get_isolation_report()
        assert report["isolation_maintained"] is True
        assert len(report["violations"]) == 0
    
    def test_provider_isolation_violation(self):
        """Test provider isolation violation detection."""
        validator = ProviderIsolationValidator()
        
        # Register same provider for paraphrase first
        validator.register_paraphrase_provider("same_provider")
        
        # Attempt to register same provider for evaluation should fail
        with pytest.raises(ValidationError) as excinfo:
            validator.register_evaluation_provider("same_provider")
        
        assert "provider_isolation_violation" in str(excinfo.value)
        assert "CRITICAL VIOLATION" in str(excinfo.value)
        
        # Check violation recorded
        report = validator.get_isolation_report()
        assert report["isolation_maintained"] is False
        assert len(report["violations"]) > 0
    
    def test_config_validation(self):
        """Test configuration validation for provider isolation."""
        validator = ProviderIsolationValidator()
        
        # Valid config with separate providers
        valid_config = {
            "transforms": [
                {"kind": "paraphrase", "provider": "hosted_heldout"}
            ],
            "models": {
                "provider_groups": [
                    {"provider": "ollama_local"}
                ]
            }
        }
        
        report = validator.validate_config(valid_config)
        assert report["valid"] is True
        assert len(report["violations"]) == 0
        
        # Invalid config with same provider
        invalid_config = {
            "transforms": [
                {"kind": "paraphrase", "provider": "same_provider"}
            ],
            "models": {
                "provider_groups": [
                    {"provider": "same_provider"}
                ]
            }
        }
        
        report = validator.validate_config(invalid_config)
        assert report["valid"] is False
        assert len(report["violations"]) > 0


class TestParaphrasePipeline:
    """Test core paraphrase pipeline functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as tmp:
            db_path = Path(tmp.name)
        
        database = Database(db_path)
        yield database
        
        # Cleanup
        database.close()
        db_path.unlink()
    
    @pytest.fixture
    def pipeline_config(self):
        """Standard pipeline configuration for testing."""
        return {
            "provider": "hosted_heldout",
            "n_candidates": 2,
            "semantic_sim_threshold": 0.85,
            "surface_divergence_min": 0.25,
            "temperature": 0.3,
            "bleu_threshold": 0.6
        }
    
    @pytest.fixture
    def mock_items(self):
        """Mock dataset items for testing."""
        return [
            {
                "item_id": "test_001",
                "question": "What is the capital of France?",
                "answer": "Paris"
            },
            {
                "item_id": "test_002", 
                "question": "How do you calculate the area of a circle?",
                "answer": "π × radius²"
            },
            {
                "item_id": "test_003",
                "question": "What causes photosynthesis in plants?",
                "answer": "Sunlight and chlorophyll"
            }
        ]
    
    def test_pipeline_initialization(self, temp_db, pipeline_config):
        """Test pipeline initialization with database integration."""
        pipeline = ParaphrasePipeline(pipeline_config, temp_db)
        
        assert pipeline.provider == "hosted_heldout"
        assert pipeline.n_candidates == 2
        assert pipeline.semantic_threshold == 0.85
        assert pipeline.surface_threshold == 0.25
        assert pipeline.temperature == 0.3
        assert pipeline.bleu_threshold == 0.6
        assert pipeline.use_database_cache is True
    
    @pytest.mark.asyncio
    async def test_paraphrase_generation_with_mock_model(self, temp_db, pipeline_config, mock_items):
        """Test paraphrase generation with mock model adapter."""
        pipeline = ParaphrasePipeline(pipeline_config, temp_db)
        
        # Create mock model adapter that generates quality paraphrases
        class MockModelAdapter:
            def generate(self, prompt, **kwargs):
                # Extract original question from prompt
                if "What is the capital of France?" in prompt:
                    paraphrase = "Which city serves as the capital of France?"
                elif "How do you calculate the area of a circle?" in prompt:
                    paraphrase = "What is the formula for finding a circle's area?"
                elif "What causes photosynthesis in plants?" in prompt:
                    paraphrase = "What triggers the photosynthesis process in plants?"
                else:
                    paraphrase = "Mock paraphrased version of the question"
                
                from scramblebench.llm.model_adapter import QueryResult
                return QueryResult(
                    text=paraphrase,
                    success=True,
                    response_time=0.5,
                    metadata={"mock": True}
                )
        
        pipeline.set_model_adapter(MockModelAdapter())
        
        # Generate paraphrases
        results = await pipeline.generate_paraphrase_cache(mock_items, write_cache=True)
        
        # Validate results structure
        assert "statistics" in results
        assert "quality_assessment" in results
        assert "provider_isolation" in results
        
        # Check statistics
        stats = results["statistics"]
        assert stats["total_items"] == 3
        assert stats["generated_count"] > 0
        assert stats["acceptance_rate"] > 0
        
        # Verify paraphrases were cached in database
        for item in mock_items:
            cached = temp_db.get_cached_paraphrase(item["item_id"], "hosted_heldout")
            assert cached is not None
            assert cached["accepted"] is True
            assert cached["cos_sim"] >= 0.85  # Meets semantic threshold
    
    @pytest.mark.asyncio
    async def test_semantic_similarity_validation(self, temp_db, pipeline_config):
        """Test semantic similarity validation with edge cases."""
        pipeline = ParaphrasePipeline(pipeline_config, temp_db)
        
        # Test item with very similar paraphrase (should pass semantic test)
        similar_item = {
            "item_id": "semantic_test",
            "question": "What is artificial intelligence?",
            "answer": "AI systems"
        }
        
        class SemanticMockAdapter:
            def generate(self, prompt, **kwargs):
                from scramblebench.llm.model_adapter import QueryResult
                # Generate semantically similar but surface-different paraphrase
                return QueryResult(
                    text="What does artificial intelligence mean?",
                    success=True,
                    response_time=0.3
                )
        
        pipeline.set_model_adapter(SemanticMockAdapter())
        
        results = await pipeline.generate_paraphrase_cache([similar_item], write_cache=True)
        
        # Should have high acceptance due to good semantic similarity
        assert results["statistics"]["acceptance_rate"] > 0.8
        
        # Check cached paraphrase meets criteria
        cached = temp_db.get_cached_paraphrase("semantic_test", "hosted_heldout")
        assert cached["cos_sim"] >= 0.85
    
    @pytest.mark.asyncio 
    async def test_surface_divergence_validation(self, temp_db, pipeline_config):
        """Test surface divergence validation requirements."""
        pipeline = ParaphrasePipeline(pipeline_config, temp_db)
        
        surface_item = {
            "item_id": "surface_test",
            "question": "The quick brown fox jumps over the lazy dog",
            "answer": "Pangram"
        }
        
        class SurfaceMockAdapter:
            def generate(self, prompt, **kwargs):
                from scramblebench.llm.model_adapter import QueryResult
                # Generate paraphrase with good surface divergence
                return QueryResult(
                    text="A swift brown fox leaps above the sleepy canine",
                    success=True,
                    response_time=0.4
                )
        
        pipeline.set_model_adapter(SurfaceMockAdapter())
        
        results = await pipeline.generate_paraphrase_cache([surface_item], write_cache=True)
        
        # Check surface divergence was properly evaluated
        cached = temp_db.get_cached_paraphrase("surface_test", "hosted_heldout")
        assert cached is not None
        assert cached["edit_ratio"] >= 0.25 or cached["bleu_score"] <= 0.6
    
    @pytest.mark.asyncio
    async def test_candidate_selection_logic(self, temp_db, pipeline_config):
        """Test that best candidate is selected from multiple candidates."""
        pipeline_config["n_candidates"] = 3  # Generate 3 candidates
        pipeline = ParaphrasePipeline(pipeline_config, temp_db)
        
        test_item = {
            "item_id": "candidate_test",
            "question": "How does machine learning work?",
            "answer": "Training algorithms on data"
        }
        
        class MultiCandidateMockAdapter:
            def __init__(self):
                self.call_count = 0
            
            def generate(self, prompt, **kwargs):
                from scramblebench.llm.model_adapter import QueryResult
                
                # Generate different quality candidates
                candidates = [
                    "How do machine learning systems function?",  # Good semantic + surface
                    "What is the mechanism behind machine learning?",  # Better semantic
                    "ML working process explanation needed"  # Poor quality
                ]
                
                response = candidates[self.call_count % len(candidates)]
                self.call_count += 1
                
                return QueryResult(
                    text=response,
                    success=True,
                    response_time=0.5
                )
        
        pipeline.set_model_adapter(MultiCandidateMockAdapter())
        
        results = await pipeline.generate_paraphrase_cache([test_item], write_cache=True)
        
        # Should successfully select best candidate
        assert results["statistics"]["generated_count"] == 1
        assert results["statistics"]["accepted_count"] == 1
        
        # Verify best candidate was cached
        cached = temp_db.get_cached_paraphrase("candidate_test", "hosted_heldout")
        assert cached is not None
        assert cached["accepted"] is True
    
    def test_cache_coverage_validation(self, temp_db, pipeline_config, mock_items):
        """Test cache coverage validation for datasets."""
        pipeline = ParaphrasePipeline(pipeline_config, temp_db)
        
        # Pre-populate cache with some items
        temp_db.cache_paraphrase(
            item_id="test_001",
            provider="hosted_heldout", 
            candidate_id=0,
            text="Which city is France's capital?",
            cos_sim=0.90,
            edit_ratio=0.35,
            bleu_score=0.4,
            accepted=True
        )
        
        # Validate coverage
        coverage = pipeline.validate_cache_coverage(mock_items)
        
        assert coverage["total_dataset_items"] == 3
        assert coverage["cached_items"] == 1  # Only one item cached
        assert coverage["missing_items"] == 2
        assert coverage["coverage_rate"] == 1/3
        assert "test_002" in coverage["missing_item_ids"]
        assert "test_003" in coverage["missing_item_ids"]


class TestParaphraseQualityReporter:
    """Test quality reporting functionality."""
    
    @pytest.fixture
    def sample_generation_results(self):
        """Sample generation results for testing."""
        return {
            "provider": "hosted_heldout",
            "statistics": {
                "total_items": 100,
                "generated_count": 95,
                "cached_hits": 5,
                "accepted_count": 97,
                "rejected_count": 3,
                "acceptance_rate": 0.97,
                "cache_hit_rate": 0.05
            },
            "rejection_analysis": {
                "semantic_failures": 1,
                "surface_failures": 1, 
                "both_failures": 1,
                "generation_failures": 0
            },
            "provider_isolation": {
                "isolation_maintained": True,
                "violations": []
            },
            "quality_assessment": {
                "meets_target_acceptance_rate": True,
                "target_acceptance_rate": 0.95,
                "actual_acceptance_rate": 0.97
            }
        }
    
    def test_quality_report_generation(self, sample_generation_results):
        """Test comprehensive quality report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            reporter = ParaphraseQualityReporter(Path(temp_dir))
            
            report = reporter.generate_comprehensive_report(
                sample_generation_results, 
                save_artifacts=False  # Don't save files for unit test
            )
            
            # Check report structure
            assert "report_metadata" in report
            assert "executive_summary" in report
            assert "detailed_metrics" in report
            assert "recommendations" in report
            
            # Check executive summary
            summary = report["executive_summary"]
            assert summary["acceptance_rate"] == 0.97
            assert summary["meets_academic_standards"] is True
            assert summary["provider_isolation_status"] == "maintained"
            
            # Check that high quality results in good overall score
            assert summary["overall_quality_score"] > 0.9
    
    def test_quality_assessment_with_issues(self):
        """Test quality assessment when there are issues."""
        problematic_results = {
            "provider": "problematic_provider",
            "statistics": {
                "total_items": 50,
                "generated_count": 30,
                "cached_hits": 0,
                "accepted_count": 45,  # 90% acceptance rate - below target
                "rejected_count": 5,
                "acceptance_rate": 0.90,
                "cache_hit_rate": 0.0
            },
            "rejection_analysis": {
                "semantic_failures": 1,
                "surface_failures": 2,
                "both_failures": 1,
                "generation_failures": 1  # Technical issues
            },
            "provider_isolation": {
                "isolation_maintained": False,  # Violation!
                "violations": [{"provider": "problematic_provider", "severity": "critical"}]
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            reporter = ParaphraseQualityReporter(Path(temp_dir))
            
            report = reporter.generate_comprehensive_report(
                problematic_results,
                save_artifacts=False
            )
            
            # Should identify problems
            summary = report["executive_summary"]
            assert summary["meets_academic_standards"] is False
            assert summary["provider_isolation_status"] == "violated"
            assert len(summary["primary_concerns"]) > 0
            
            # Should have immediate actions
            assert len(report["recommendations"]["immediate_actions"]) > 0
            
            # Overall score should be low due to issues
            assert summary["overall_quality_score"] < 0.8


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    def test_create_paraphrase_pipeline_factory(self):
        """Test pipeline factory function with config validation."""
        with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as tmp:
            db_path = Path(tmp.name)
        
        try:
            # Valid config
            config = {
                "run": {"seed": 1337},
                "transforms": [
                    {
                        "kind": "paraphrase",
                        "provider": "hosted_heldout",
                        "n_candidates": 2,
                        "semantic_sim_threshold": 0.85,
                        "surface_divergence_min": 0.25
                    }
                ],
                "models": {
                    "provider_groups": [
                        {"provider": "ollama_local"}  # Different provider - good!
                    ]
                }
            }
            
            pipeline = create_paraphrase_pipeline(config, db_path)
            
            assert pipeline.provider == "hosted_heldout"
            assert pipeline.n_candidates == 2
            assert pipeline.seed == 1337
            
        finally:
            db_path.unlink()
    
    def test_provider_isolation_enforcement_in_factory(self):
        """Test that factory enforces provider isolation."""
        with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as tmp:
            db_path = Path(tmp.name)
        
        try:
            # Invalid config with same provider
            config = {
                "run": {"seed": 1337},
                "transforms": [
                    {
                        "kind": "paraphrase", 
                        "provider": "same_provider"  # Same as evaluation!
                    }
                ],
                "models": {
                    "provider_groups": [
                        {"provider": "same_provider"}  # Violation!
                    ]
                }
            }
            
            with pytest.raises(ValidationError) as excinfo:
                create_paraphrase_pipeline(config, db_path)
            
            assert "provider_isolation_violation" in str(excinfo.value)
            
        finally:
            db_path.unlink()


class TestAcademicStandards:
    """Test compliance with academic standards."""
    
    def test_95_percent_acceptance_rate_target(self):
        """Test that 95% acceptance rate target is properly enforced."""
        # Results that meet the target
        good_results = {
            "statistics": {"acceptance_rate": 0.96, "total_items": 100},
            "quality_assessment": {"meets_target_acceptance_rate": True}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            reporter = ParaphraseQualityReporter(Path(temp_dir))
            report = reporter.generate_comprehensive_report(good_results, save_artifacts=False)
            
            assert report["executive_summary"]["meets_academic_standards"] is True
            assert len([rec for rec in report["recommendations"]["immediate_actions"] 
                       if "acceptance rate" in rec.lower()]) == 0
        
        # Results that don't meet the target
        bad_results = {
            "statistics": {"acceptance_rate": 0.92, "total_items": 100},
            "quality_assessment": {"meets_target_acceptance_rate": False},
            "rejection_analysis": {"semantic_failures": 8},
            "provider_isolation": {"isolation_maintained": True}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            reporter = ParaphraseQualityReporter(Path(temp_dir))
            report = reporter.generate_comprehensive_report(bad_results, save_artifacts=False)
            
            assert report["executive_summary"]["meets_academic_standards"] is False
            assert len([rec for rec in report["recommendations"]["immediate_actions"] 
                       if "acceptance rate" in rec.lower()]) > 0
    
    def test_provider_isolation_academic_requirement(self):
        """Test that provider isolation is treated as academic requirement."""
        violation_results = {
            "statistics": {"acceptance_rate": 0.98, "total_items": 100},
            "provider_isolation": {
                "isolation_maintained": False,
                "violations": [{"provider": "contaminated", "severity": "critical"}]
            },
            "rejection_analysis": {}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            reporter = ParaphraseQualityReporter(Path(temp_dir))
            report = reporter.generate_comprehensive_report(violation_results, save_artifacts=False)
            
            # Should fail academic standards due to provider violation
            assert report["executive_summary"]["meets_academic_standards"] is False
            assert report["executive_summary"]["provider_isolation_status"] == "violated"
            
            # Should have critical recommendations
            immediate_actions = report["recommendations"]["immediate_actions"]
            assert len([rec for rec in immediate_actions if "CRITICAL" in rec]) > 0


if __name__ == "__main__":
    # Run basic smoke test
    pytest.main([__file__, "-v"])