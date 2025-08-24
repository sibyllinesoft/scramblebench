"""
Comprehensive Paraphrase Pipeline for ScrambleBench Step S1 Implementation.

This module implements the complete paraphrase control pipeline with provider isolation
enforcement, database-integrated caching, quality control checks, and academic-grade
reporting as specified in TODO.md step S1_paraphrase_pipeline.

Academic Requirements:
- Provider isolation to prevent contamination between paraphrase generation and evaluation
- High-quality paraphrases with semantic equivalence (≥0.85) and surface divergence (≥0.25)
- Immutable caching system with DuckDB integration
- Target ≥95% acceptance rate for generated paraphrases
- Comprehensive quality reporting and rejection analysis

Implementation Notes:
- Temperature=0.3 ONLY for paraphrase generation, never for evaluation
- n=2 candidates per item with best-candidate selection
- Both semantic similarity AND surface divergence checks must pass
- Zero evaluation calls should use the paraphrase provider (enforced validation)
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from scramblebench.core.database import Database
from scramblebench.core.exceptions import ValidationError, ModelError
from scramblebench.llm.model_adapter import ModelAdapter, QueryResult
from scramblebench.transforms.paraphrase import ParaphraseValidator, ParaphraseTransform
from scramblebench.transforms.base import TransformResult
from scramblebench.transforms.paraphrase_errors import (
    AcademicLogger, ParaphraseErrorHandler, ParaphraseError,
    ParaphraseErrorCode, ParaphraseErrorSeverity,
    get_academic_logger, get_error_handler
)

logger = get_academic_logger(__name__)


class ProviderIsolationValidator:
    """Validates that paraphrase providers are never used for evaluation."""
    
    def __init__(self):
        self.paraphrase_providers: Set[str] = set()
        self.evaluation_providers: Set[str] = set()
        self.violations: List[Dict[str, Any]] = []
    
    def register_paraphrase_provider(self, provider: str) -> None:
        """Register a provider as being used for paraphrase generation."""
        if provider in self.evaluation_providers:
            error_handler = get_error_handler()
            error = error_handler.handle_provider_isolation_violation(
                provider,
                {"operation": "paraphrase_registration", "existing_eval_providers": list(self.evaluation_providers)}
            )
            raise ValidationError(str(error), error_code=error.error_code.value)
        
        self.paraphrase_providers.add(provider)
        logger.info(f"Registered paraphrase provider: {provider}", extra={
            "provider": provider,
            "total_paraphrase_providers": len(self.paraphrase_providers)
        })
    
    def register_evaluation_provider(self, provider: str) -> None:
        """Register a provider as being used for evaluation."""
        if provider in self.paraphrase_providers:
            error_handler = get_error_handler()
            error = error_handler.handle_provider_isolation_violation(
                provider,
                {
                    "operation": "evaluation_registration",
                    "existing_paraphrase_providers": list(self.paraphrase_providers),
                    "academic_impact": "critical"
                }
            )
            
            violation = {
                "timestamp": datetime.now().isoformat(),
                "provider": provider,
                "violation_type": "evaluation_provider_used_for_paraphrase",
                "severity": "critical"
            }
            self.violations.append(violation)
            
            raise ValidationError(str(error), error_code=error.error_code.value, context=violation)
        
        self.evaluation_providers.add(provider)
        logger.info(f"Registered evaluation provider: {provider}", extra={
            "provider": provider,
            "total_evaluation_providers": len(self.evaluation_providers)
        })
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration for provider isolation."""
        validation_report = {
            "valid": True,
            "paraphrase_providers": [],
            "evaluation_providers": [],
            "violations": [],
            "recommendations": []
        }
        
        # Extract paraphrase providers
        if "transforms" in config:
            for transform in config["transforms"]:
                if transform.get("kind") == "paraphrase":
                    provider = transform.get("provider")
                    if provider:
                        validation_report["paraphrase_providers"].append(provider)
        
        # Extract evaluation providers
        if "models" in config and "provider_groups" in config["models"]:
            for group in config["models"]["provider_groups"]:
                provider = group.get("provider")
                if provider:
                    validation_report["evaluation_providers"].append(provider)
        
        # Check for overlaps
        paraphrase_set = set(validation_report["paraphrase_providers"])
        evaluation_set = set(validation_report["evaluation_providers"])
        overlaps = paraphrase_set.intersection(evaluation_set)
        
        if overlaps:
            validation_report["valid"] = False
            for provider in overlaps:
                violation = {
                    "provider": provider,
                    "issue": "Provider used for both paraphrase generation and evaluation",
                    "severity": "critical"
                }
                validation_report["violations"].append(violation)
        
        # Add recommendations
        if not validation_report["paraphrase_providers"]:
            validation_report["recommendations"].append(
                "No paraphrase providers configured. Consider adding a held-out provider."
            )
        
        if validation_report["paraphrase_providers"] == validation_report["evaluation_providers"]:
            validation_report["recommendations"].append(
                "Consider using different model families/providers for paraphrase vs evaluation."
            )
        
        return validation_report
    
    def get_isolation_report(self) -> Dict[str, Any]:
        """Generate comprehensive provider isolation report."""
        return {
            "paraphrase_providers": list(self.paraphrase_providers),
            "evaluation_providers": list(self.evaluation_providers),
            "isolation_maintained": len(self.paraphrase_providers.intersection(self.evaluation_providers)) == 0,
            "violations": self.violations,
            "total_violations": len(self.violations)
        }


class ParaphrasePipeline:
    """
    Complete paraphrase control pipeline for contamination detection.
    
    This pipeline implements Step S1 from TODO.md with academic-grade quality control,
    provider isolation enforcement, and comprehensive reporting capabilities.
    """
    
    def __init__(self, 
                 config: Dict[str, Any], 
                 database: Database,
                 seed: int = 1337):
        """
        Initialize the paraphrase pipeline.
        
        Args:
            config: Paraphrase configuration dict
            database: Database instance for caching
            seed: Random seed for reproducibility
        """
        self.config = config
        self.database = database
        self.seed = seed
        
        # Core components
        self.validator = ParaphraseValidator()
        self.isolation_validator = ProviderIsolationValidator()
        self.model_adapter = ModelAdapter(
            timeout=config.get("timeout", 60),
            max_retries=config.get("max_retries", 3),
            validate_responses=True
        )
        
        # Configuration parameters
        self.provider = config.get("provider", "hosted_heldout")
        self.n_candidates = config.get("n_candidates", 2)
        self.semantic_threshold = config.get("semantic_sim_threshold", 0.85)
        self.surface_threshold = config.get("surface_divergence_min", 0.25)
        self.temperature = config.get("temperature", 0.3)  # Only for generation
        
        # Quality control parameters
        self.bleu_threshold = config.get("bleu_threshold", 0.6)  # BLEU ≤ 0.6 for surface divergence
        
        # Register paraphrase provider for isolation
        self.isolation_validator.register_paraphrase_provider(self.provider)
        
        # Initialize error handler
        self.error_handler = get_error_handler()
        
        # Statistics tracking
        self.stats = {
            "total_items": 0,
            "generated_count": 0,
            "cached_hits": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "rejection_reasons": {
                "semantic_fail": 0,
                "surface_fail": 0,
                "both_fail": 0,
                "generation_fail": 0
            }
        }
        
        logger.info(f"ParaphrasePipeline initialized with provider: {self.provider}", extra={
            "provider": self.provider,
            "n_candidates": self.n_candidates,
            "semantic_threshold": self.semantic_threshold,
            "surface_threshold": self.surface_threshold,
            "temperature": self.temperature,
            "seed": self.seed
        })
    
    def set_model_adapter(self, adapter: Any) -> None:
        """Set the model adapter for paraphrase generation."""
        self.model_adapter_instance = adapter
    
    async def generate_paraphrase_cache(self, 
                                       items: List[Dict[str, Any]], 
                                       write_cache: bool = True) -> Dict[str, Any]:
        """
        Generate paraphrases for a list of items and cache them in the database.
        
        Args:
            items: List of items with 'item_id', 'question', and 'answer'
            write_cache: Whether to write results to the database cache
            
        Returns:
            Generation report with statistics and quality metrics
        """
        self.stats["total_items"] = len(items)
        
        logger.info(f"Starting paraphrase generation for {len(items)} items")
        logger.info(f"Provider: {self.provider}, n_candidates: {self.n_candidates}")
        logger.info(f"Semantic threshold: {self.semantic_threshold}, Surface threshold: {self.surface_threshold}")
        
        results = []
        
        # Process items with concurrency control
        semaphore = asyncio.Semaphore(4)  # Limit concurrent requests
        
        tasks = []
        for item in items:
            task = self._process_item_with_semaphore(semaphore, item, write_cache)
            tasks.append(task)
        
        # Execute all tasks
        item_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        for i, result in enumerate(item_results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process item {items[i]['item_id']}: {result}")
                self.stats["rejected_count"] += 1
                self.stats["rejection_reasons"]["generation_fail"] += 1
            else:
                results.append(result)
        
        # Generate comprehensive report
        report = self._generate_generation_report(results)
        
        # Log final quality metrics for academic analysis
        logger.log_quality_metrics({
            "acceptance_rate": report.get("statistics", {}).get("acceptance_rate", 0.0),
            "total_items": report.get("statistics", {}).get("total_items", 0),
            "cache_hit_rate": report.get("statistics", {}).get("cache_hit_rate", 0.0)
        })
        
        # Check if acceptance rate meets academic standards
        acceptance_rate = report.get("statistics", {}).get("acceptance_rate", 0.0)
        if acceptance_rate < 0.95:
            self.error_handler.handle_acceptance_rate_below_target(
                acceptance_rate,
                context={
                    "total_items": self.stats["total_items"],
                    "rejection_breakdown": self.stats["rejection_reasons"]
                }
            )
        
        logger.info(f"Paraphrase generation completed: {acceptance_rate:.2%} acceptance rate", extra={
            "acceptance_rate": acceptance_rate,
            "meets_academic_standard": acceptance_rate >= 0.95,
            "total_processed": self.stats["total_items"],
            "session_summary": logger.get_session_summary()
        })
        
        return report
    
    async def _process_item_with_semaphore(self, 
                                         semaphore: asyncio.Semaphore,
                                         item: Dict[str, Any], 
                                         write_cache: bool) -> Dict[str, Any]:
        """Process a single item with concurrency control."""
        async with semaphore:
            return await self._process_single_item(item, write_cache)
    
    async def _process_single_item(self, item: Dict[str, Any], write_cache: bool) -> Dict[str, Any]:
        """
        Process a single item for paraphrase generation.
        
        Args:
            item: Item dictionary with 'item_id', 'question', 'answer'
            write_cache: Whether to write to database cache
            
        Returns:
            Processing result dictionary
        """
        item_id = item["item_id"]
        question = item["question"]
        
        # Check cache first
        cached_paraphrase = self.database.get_cached_paraphrase(item_id, self.provider)
        
        if cached_paraphrase and cached_paraphrase["accepted"]:
            self.stats["cached_hits"] += 1
            self.stats["accepted_count"] += 1
            
            return {
                "item_id": item_id,
                "original_text": question,
                "paraphrase_text": cached_paraphrase["text"],
                "cached": True,
                "accepted": True,
                "semantic_score": cached_paraphrase["cos_sim"],
                "surface_divergent": True,  # Cached items were already validated
                "generation_metadata": {}
            }
        
        # Generate new paraphrase candidates
        try:
            candidates = await self._generate_candidates(question)
            
            if not candidates:
                self.stats["rejected_count"] += 1
                self.stats["rejection_reasons"]["generation_fail"] += 1
                return {
                    "item_id": item_id,
                    "original_text": question,
                    "cached": False,
                    "accepted": False,
                    "rejection_reason": "No candidates generated"
                }
            
            # Select best candidate
            best_candidate = self._select_best_candidate(question, candidates)
            
            if best_candidate is None:
                # Log rejection reasons
                for candidate in candidates:
                    validation = candidate.get("validation", {})
                    if not validation.get("semantic_valid", False) and not validation.get("surface_divergent", False):
                        self.stats["rejection_reasons"]["both_fail"] += 1
                    elif not validation.get("semantic_valid", False):
                        self.stats["rejection_reasons"]["semantic_fail"] += 1
                    elif not validation.get("surface_divergent", False):
                        self.stats["rejection_reasons"]["surface_fail"] += 1
                
                self.stats["rejected_count"] += 1
                
                return {
                    "item_id": item_id,
                    "original_text": question,
                    "cached": False,
                    "accepted": False,
                    "rejection_reason": "No candidates passed validation",
                    "candidates_generated": len(candidates),
                    "failed_candidates": candidates
                }
            
            # Cache successful paraphrase in database
            if write_cache:
                try:
                    validation = best_candidate["validation"]
                    self.database.cache_paraphrase(
                        item_id=item_id,
                        provider=self.provider,
                        candidate_id=best_candidate["candidate_id"],
                        text=best_candidate["text"],
                        cos_sim=validation["semantic_score"],
                        edit_ratio=validation["edit_ratio"],
                        bleu_score=validation["bleu_score"],
                        accepted=True
                    )
                    
                    logger.log_paraphrase_event(
                        event_type="cache_write",
                        item_id=item_id,
                        success=True,
                        details={"provider": self.provider, "semantic_score": validation["semantic_score"]}
                    )
                except Exception as cache_error:
                    # Handle cache failures gracefully
                    error = self.error_handler.handle_cache_failure("write", item_id, cache_error)
                    logger.warning(f"Cache write failed but paraphrase generation succeeded: {item_id}")
                
                # Also cache rejected candidates for analysis
                for i, candidate in enumerate(candidates):
                    if candidate["candidate_id"] != best_candidate["candidate_id"]:
                        validation = candidate.get("validation", {})
                        self.database.cache_paraphrase(
                            item_id=item_id,
                            provider=self.provider,
                            candidate_id=candidate["candidate_id"],
                            text=candidate["text"],
                            cos_sim=validation.get("semantic_score", 0.0),
                            edit_ratio=validation.get("edit_ratio", 0.0),
                            bleu_score=validation.get("bleu_score", 0.0),
                            accepted=False
                        )
            
            self.stats["generated_count"] += 1
            self.stats["accepted_count"] += 1
            
            return {
                "item_id": item_id,
                "original_text": question,
                "paraphrase_text": best_candidate["text"],
                "cached": False,
                "accepted": True,
                "semantic_score": best_candidate["validation"]["semantic_score"],
                "surface_divergent": best_candidate["validation"]["surface_divergent"],
                "edit_ratio": best_candidate["validation"]["edit_ratio"],
                "bleu_score": best_candidate["validation"]["bleu_score"],
                "candidates_generated": len(candidates),
                "generation_metadata": best_candidate.get("generation_metadata", {})
            }
            
        except Exception as e:
            # Use academic error handler
            error = self.error_handler.handle_generation_failure(item_id, e)
            self.stats["rejected_count"] += 1
            self.stats["rejection_reasons"]["generation_fail"] += 1
            
            return {
                "item_id": item_id,
                "original_text": question,
                "cached": False,
                "accepted": False,
                "rejection_reason": f"Processing error: {str(e)}",
                "error_details": error.to_dict()
            }
    
    async def _generate_candidates(self, text: str) -> List[Dict[str, Any]]:
        """Generate paraphrase candidates using the model adapter."""
        if not hasattr(self, 'model_adapter_instance'):
            raise ModelError("Model adapter not set. Use set_model_adapter() first.")
        
        candidates = []
        prompt = self._create_paraphrase_prompt(text)
        
        # Generate multiple candidates with different seeds
        for i in range(self.n_candidates):
            try:
                # Use async query if supported
                if hasattr(self.model_adapter_instance, 'generate'):
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.model_adapter_instance.generate(
                            prompt=prompt,
                            temperature=self.temperature,  # Non-zero for diversity
                            max_tokens=len(text) * 2,
                            seed=self.seed + i
                        )
                    )
                else:
                    # Fallback to query method
                    result = self.model_adapter.query(
                        self.model_adapter_instance,
                        prompt,
                        temperature=self.temperature,
                        max_tokens=len(text) * 2,
                        seed=self.seed + i
                    )
                
                if result.success and result.text.strip():
                    paraphrase_text = self._extract_paraphrase_from_response(result.text)
                    
                    if paraphrase_text and paraphrase_text.strip() != text.strip():
                        candidates.append({
                            "text": paraphrase_text,
                            "candidate_id": i,
                            "generation_metadata": {
                                "response_time": result.response_time,
                                "raw_response": result.text[:200] + "..." if len(result.text) > 200 else result.text
                            }
                        })
                
            except Exception as e:
                logger.warning(f"Failed to generate paraphrase candidate {i}: {e}")
                continue
        
        return candidates
    
    def _create_paraphrase_prompt(self, text: str) -> str:
        """Create optimized prompt for paraphrase generation."""
        return f"""Please rewrite the following text to have exactly the same meaning but use different words and sentence structure. The rewritten version must:

1. Preserve the exact same meaning and information
2. Use different vocabulary and phrasing
3. Maintain the same level of complexity

Original text: {text}

Rewritten text:"""
    
    def _extract_paraphrase_from_response(self, response: str) -> Optional[str]:
        """Extract paraphrase text from model response."""
        lines = response.strip().split('\n')
        
        # Look for the rewritten text after common indicators
        for i, line in enumerate(lines):
            cleaned = line.strip()
            if not cleaned:
                continue
            
            # Skip common prefixes
            if any(cleaned.lower().startswith(prefix) for prefix in [
                'original:', 'rewritten:', 'text:', 'paraphrase:', 'here is', 'the rewritten'
            ]):
                # If this line has content after the prefix, use it
                parts = cleaned.split(':', 1)
                if len(parts) > 1 and parts[1].strip():
                    return parts[1].strip()
                # Otherwise, use the next non-empty line
                continue
            
            # This looks like the actual paraphrase
            return cleaned
        
        return None
    
    def _select_best_candidate(self, original: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select the best paraphrase candidate based on validation criteria."""
        validated_candidates = []
        
        for candidate in candidates:
            validation = self.validator.validate_paraphrase(
                original, candidate["text"],
                semantic_threshold=self.semantic_threshold,
                surface_threshold=self.surface_threshold
            )
            
            # Update validation to use BLEU threshold
            if validation["bleu_score"] <= self.bleu_threshold:
                validation["surface_divergent"] = True
            
            candidate["validation"] = validation
            
            if validation["overall_valid"]:
                validated_candidates.append(candidate)
        
        if not validated_candidates:
            return None
        
        # Select candidate with highest semantic score among valid ones
        best_candidate = max(validated_candidates, 
                           key=lambda c: c["validation"]["semantic_score"])
        
        return best_candidate
    
    def _generate_generation_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive generation report."""
        total_items = self.stats["total_items"]
        accepted_count = self.stats["accepted_count"]
        
        acceptance_rate = accepted_count / total_items if total_items > 0 else 0.0
        
        report = {
            "generation_timestamp": datetime.now().isoformat(),
            "provider": self.provider,
            "configuration": {
                "n_candidates": self.n_candidates,
                "semantic_threshold": self.semantic_threshold,
                "surface_threshold": self.surface_threshold,
                "bleu_threshold": self.bleu_threshold,
                "temperature": self.temperature,
                "seed": self.seed
            },
            "statistics": {
                "total_items": total_items,
                "generated_count": self.stats["generated_count"],
                "cached_hits": self.stats["cached_hits"],
                "accepted_count": accepted_count,
                "rejected_count": self.stats["rejected_count"],
                "acceptance_rate": acceptance_rate,
                "cache_hit_rate": self.stats["cached_hits"] / total_items if total_items > 0 else 0.0
            },
            "rejection_analysis": {
                "semantic_failures": self.stats["rejection_reasons"]["semantic_fail"],
                "surface_failures": self.stats["rejection_reasons"]["surface_fail"],
                "both_failures": self.stats["rejection_reasons"]["both_fail"],
                "generation_failures": self.stats["rejection_reasons"]["generation_fail"]
            },
            "quality_assessment": {
                "meets_target_acceptance_rate": acceptance_rate >= 0.95,
                "target_acceptance_rate": 0.95,
                "actual_acceptance_rate": acceptance_rate
            },
            "provider_isolation": self.isolation_validator.get_isolation_report(),
            "recommendation": self._generate_recommendations(acceptance_rate)
        }
        
        return report
    
    def _generate_recommendations(self, acceptance_rate: float) -> List[str]:
        """Generate recommendations based on acceptance rate and other metrics."""
        recommendations = []
        
        if acceptance_rate < 0.95:
            recommendations.append(
                f"Acceptance rate ({acceptance_rate:.2%}) below target (95%). "
                f"Consider adjusting thresholds or improving prompt engineering."
            )
        
        if acceptance_rate < 0.80:
            recommendations.append(
                "Very low acceptance rate. Consider using a more capable model for paraphrase generation."
            )
        
        semantic_fails = self.stats["rejection_reasons"]["semantic_fail"]
        surface_fails = self.stats["rejection_reasons"]["surface_fail"]
        total_rejects = self.stats["rejected_count"]
        
        if total_rejects > 0:
            if semantic_fails / total_rejects > 0.5:
                recommendations.append(
                    "High semantic failure rate. Consider lowering semantic similarity threshold."
                )
            
            if surface_fails / total_rejects > 0.5:
                recommendations.append(
                    "High surface divergence failure rate. Consider lowering surface divergence threshold."
                )
        
        if self.stats["generated_count"] == 0:
            recommendations.append(
                "No paraphrases generated successfully. Check model connectivity and prompt effectiveness."
            )
        
        return recommendations
    
    def validate_cache_coverage(self, dataset_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate paraphrase cache coverage for a dataset."""
        coverage_stats = self.database.get_paraphrase_coverage(self.provider)
        
        # Calculate coverage per item
        item_coverage = []
        for item in dataset_items:
            cached = self.database.get_cached_paraphrase(item["item_id"], self.provider)
            item_coverage.append({
                "item_id": item["item_id"],
                "has_paraphrase": cached is not None and cached["accepted"],
                "semantic_score": cached["cos_sim"] if cached else None
            })
        
        missing_items = [item for item in item_coverage if not item["has_paraphrase"]]
        
        return {
            "provider": self.provider,
            "total_dataset_items": len(dataset_items),
            "cached_items": len([item for item in item_coverage if item["has_paraphrase"]]),
            "missing_items": len(missing_items),
            "coverage_rate": (len(dataset_items) - len(missing_items)) / len(dataset_items) if dataset_items else 0.0,
            "database_stats": coverage_stats,
            "missing_item_ids": [item["item_id"] for item in missing_items]
        }


def create_paraphrase_pipeline(config: Dict[str, Any], database_path: Path) -> ParaphrasePipeline:
    """
    Factory function to create a configured paraphrase pipeline.
    
    Args:
        config: Configuration dictionary from YAML
        database_path: Path to DuckDB database
        
    Returns:
        Configured ParaphrasePipeline instance
    """
    database = Database(database_path)
    
    # Extract paraphrase config
    paraphrase_config = None
    for transform in config.get("transforms", []):
        if transform.get("kind") == "paraphrase":
            paraphrase_config = transform
            break
    
    if not paraphrase_config:
        raise ValidationError("No paraphrase transform configuration found")
    
    # Validate provider isolation
    pipeline = ParaphrasePipeline(
        config=paraphrase_config,
        database=database,
        seed=config.get("run", {}).get("seed", 1337)
    )
    
    # Validate full configuration for provider isolation
    isolation_report = pipeline.isolation_validator.validate_config(config)
    if not isolation_report["valid"]:
        raise ValidationError(
            f"Provider isolation validation failed: {isolation_report['violations']}",
            error_code="provider_isolation_violation",
            context=isolation_report
        )
    
    return pipeline