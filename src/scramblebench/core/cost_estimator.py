"""
Cost estimation system for ScrambleBench evaluations.

Provides token counting, API pricing integration, and budget cap enforcement
to prevent cost overruns during large-scale academic research workloads.
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import tiktoken

from .unified_config import ScrambleBenchConfig, ModelConfig
from .adapters import create_adapter, BaseModelAdapter


logger = logging.getLogger(__name__)


@dataclass
class TokenCostEstimate:
    """Token cost estimate for a single evaluation."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    provider: str
    model: str


@dataclass
class EvaluationCostProjection:
    """Complete cost projection for evaluation run."""
    total_evaluations: int
    total_tokens: int
    total_cost_usd: float
    breakdown_by_model: Dict[str, TokenCostEstimate]
    breakdown_by_transform: Dict[str, float]
    breakdown_by_dataset: Dict[str, float]
    within_budget: bool
    budget_limit: float
    projection_confidence: float  # 0.0 to 1.0


class CostEstimator:
    """Estimates costs for ScrambleBench evaluations with token counting."""
    
    # API pricing per 1M tokens (as of 2024-2025)
    PRICING_TABLE = {
        "openrouter": {
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "llama-3.1-70b": {"input": 0.88, "output": 0.88},
            "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
            "default": {"input": 1.00, "output": 3.00}  # Fallback pricing
        },
        "ollama": {
            "default": {"input": 0.0, "output": 0.0}  # Local models are free
        },
        "anthropic": {
            "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
            "claude-3-haiku": {"input": 0.25, "output": 1.25}
        },
        "openai": {
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4o": {"input": 2.50, "output": 10.00}
        }
    }
    
    def __init__(self, config: ScrambleBenchConfig):
        self.config = config
        self.tokenizers: Dict[str, Any] = {}
        self._load_tokenizers()
    
    def _load_tokenizers(self):
        """Load tokenizers for cost estimation."""
        # Load common tokenizers
        try:
            self.tokenizers["gpt-4"] = tiktoken.encoding_for_model("gpt-4")
            self.tokenizers["gpt-3.5-turbo"] = tiktoken.encoding_for_model("gpt-3.5-turbo")
            self.tokenizers["default"] = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load some tokenizers: {e}")
            # Fallback to character-based estimation
            self.tokenizers["default"] = None
    
    def get_tokenizer_for_model(self, model_id: str) -> Any:
        """Get appropriate tokenizer for model."""
        # Map model IDs to tokenizers
        if "gpt-4" in model_id.lower():
            return self.tokenizers.get("gpt-4", self.tokenizers["default"])
        elif "gpt-3.5" in model_id.lower() or "gpt-35" in model_id.lower():
            return self.tokenizers.get("gpt-3.5-turbo", self.tokenizers["default"])
        else:
            return self.tokenizers["default"]
    
    def count_tokens(self, text: str, model_id: str = "default") -> int:
        """Count tokens in text using appropriate tokenizer."""
        tokenizer = self.get_tokenizer_for_model(model_id)
        
        if tokenizer is None:
            # Character-based fallback (rough estimate: ~4 chars per token)
            return max(1, len(text) // 4)
        
        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Tokenization failed for {model_id}: {e}")
            # Fallback to character-based estimation
            return max(1, len(text) // 4)
    
    def get_model_pricing(self, provider: str, model_id: str) -> Dict[str, float]:
        """Get pricing information for model."""
        provider_pricing = self.PRICING_TABLE.get(provider, {})
        
        # Try exact match first
        if model_id in provider_pricing:
            return provider_pricing[model_id]
        
        # Try partial matches for variants
        for price_model, pricing in provider_pricing.items():
            if price_model in model_id or model_id in price_model:
                return pricing
        
        # Fallback to provider default
        if "default" in provider_pricing:
            return provider_pricing["default"]
        
        # Ultimate fallback
        logger.warning(f"No pricing found for {provider}/{model_id}, using conservative estimate")
        return {"input": 2.00, "output": 8.00}
    
    def estimate_single_evaluation_cost(
        self, 
        prompt: str, 
        expected_response_tokens: int,
        model_config: ModelConfig
    ) -> TokenCostEstimate:
        """Estimate cost for single evaluation."""
        prompt_tokens = self.count_tokens(prompt, model_config.name)
        completion_tokens = expected_response_tokens
        total_tokens = prompt_tokens + completion_tokens
        
        pricing = self.get_model_pricing(model_config.provider, model_config.name)
        
        # Calculate cost (pricing is per 1M tokens)
        prompt_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        completion_cost = (completion_tokens / 1_000_000) * pricing["output"]
        total_cost = prompt_cost + completion_cost
        
        return TokenCostEstimate(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=total_cost,
            provider=model_config.provider,
            model=model_config.name
        )
    
    async def project_evaluation_costs(
        self, 
        sample_prompts: List[str],
        expected_response_tokens: int = 256
    ) -> EvaluationCostProjection:
        """Project total costs for evaluation run."""
        logger.info("Projecting evaluation costs...")
        
        # Calculate total evaluation count
        total_models = sum(len(pg.list) for pg in self.config.models.provider_groups)
        total_datasets = len(self.config.datasets)
        
        # Count transform evaluations
        transform_multiplier = 0
        transform_costs = {}
        
        for transform in self.config.transforms:
            if transform.kind == "original":
                multiplier = 1
                transform_costs["original"] = 0
            elif transform.kind == "paraphrase":
                multiplier = 1  # Same cost as original
                # Add paraphrase generation cost (separate provider)
                paraphrase_gen_cost = 0.1  # Estimated per item
                transform_costs["paraphrase"] = paraphrase_gen_cost
            elif transform.kind == "scramble":
                multiplier = len(transform.levels)
                transform_costs["scramble"] = 0
            
            transform_multiplier += multiplier
        
        # Calculate sample size
        avg_sample_size = sum(ds.sample_size for ds in self.config.datasets) / len(self.config.datasets)
        
        total_evaluations = int(total_models * total_datasets * transform_multiplier * avg_sample_size)
        
        # Use sample prompts to estimate average cost
        model_costs = {}
        total_cost = 0.0
        total_tokens = 0
        
        all_models = self.config.get_all_models()
        sample_prompt = sample_prompts[0] if sample_prompts else "Q: What is 2+2?\nA:"
        
        for model_config in all_models:
            # Estimate cost per evaluation for this model
            cost_estimate = self.estimate_single_evaluation_cost(
                sample_prompt, expected_response_tokens, model_config
            )
            
            # Calculate total cost for this model across all evaluations
            evaluations_per_model = total_evaluations // total_models
            model_total_cost = cost_estimate.cost_usd * evaluations_per_model
            
            model_costs[model_config.name] = cost_estimate
            total_cost += model_total_cost
            total_tokens += cost_estimate.total_tokens * evaluations_per_model
        
        # Calculate dataset breakdown (assume equal distribution)
        dataset_costs = {
            ds.name: total_cost / len(self.config.datasets) 
            for ds in self.config.datasets
        }
        
        # Add transform generation costs
        transform_gen_costs = {}
        for transform_type, per_item_cost in transform_costs.items():
            if per_item_cost > 0:
                total_items = sum(ds.sample_size for ds in self.config.datasets)
                transform_gen_costs[transform_type] = per_item_cost * total_items
                total_cost += transform_gen_costs[transform_type]
        
        # Combine transform costs
        all_transform_costs = {**transform_costs, **transform_gen_costs}
        
        # Check budget
        within_budget = total_cost <= self.config.run.max_cost_usd
        
        # Calculate confidence (based on sample size and pricing availability)
        confidence = 0.8  # Base confidence
        if len(sample_prompts) >= 10:
            confidence += 0.1
        if all(self.get_model_pricing(m.provider, m.name) != {"input": 2.00, "output": 8.00} 
               for m in all_models):
            confidence += 0.1
        
        projection = EvaluationCostProjection(
            total_evaluations=total_evaluations,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            breakdown_by_model=model_costs,
            breakdown_by_transform=all_transform_costs,
            breakdown_by_dataset=dataset_costs,
            within_budget=within_budget,
            budget_limit=self.config.run.max_cost_usd,
            projection_confidence=min(1.0, confidence)
        )
        
        logger.info(f"Cost projection: ${total_cost:.2f} for {total_evaluations} evaluations")
        logger.info(f"Budget limit: ${self.config.run.max_cost_usd:.2f}")
        logger.info(f"Within budget: {within_budget}")
        
        return projection
    
    def validate_budget(self, projection: EvaluationCostProjection) -> Tuple[bool, List[str]]:
        """Validate projected costs against budget with detailed warnings."""
        warnings = []
        
        if not projection.within_budget:
            overage = projection.total_cost_usd - projection.budget_limit
            overage_pct = (overage / projection.budget_limit) * 100
            
            warnings.append(
                f"BUDGET EXCEEDED: Projected cost ${projection.total_cost_usd:.2f} "
                f"exceeds limit ${projection.budget_limit:.2f} by ${overage:.2f} ({overage_pct:.1f}%)"
            )
        
        # Warn about low confidence projections
        if projection.projection_confidence < 0.7:
            warnings.append(
                f"LOW CONFIDENCE: Cost projection confidence is {projection.projection_confidence:.1%}. "
                "Consider running with more sample data or verifying pricing information."
            )
        
        # Warn about expensive models
        for model_name, estimate in projection.breakdown_by_model.items():
            if estimate.cost_usd > 0.01:  # More than 1 cent per evaluation
                total_model_cost = estimate.cost_usd * (projection.total_evaluations / len(projection.breakdown_by_model))
                if total_model_cost > projection.budget_limit * 0.3:  # More than 30% of budget
                    warnings.append(
                        f"EXPENSIVE MODEL: {model_name} projected to cost ${total_model_cost:.2f} "
                        f"({total_model_cost/projection.total_cost_usd:.1%} of total budget)"
                    )
        
        return projection.within_budget, warnings
    
    async def enforce_budget_cap(self, projection: EvaluationCostProjection) -> bool:
        """Enforce hard budget cap - raises exception if exceeded."""
        within_budget, warnings = self.validate_budget(projection)
        
        # Log all warnings
        for warning in warnings:
            logger.warning(warning)
        
        if not within_budget:
            raise ValueError(
                f"HARD BUDGET CAP EXCEEDED: Projected cost ${projection.total_cost_usd:.2f} "
                f"exceeds maximum allowed ${projection.budget_limit:.2f}. "
                f"Reduce sample sizes, models, or increase budget limit."
            )
        
        return True
    
    def export_cost_projection(self, projection: EvaluationCostProjection, output_path: Path):
        """Export cost projection to JSON file."""
        # Convert to serializable format
        export_data = {
            "total_evaluations": projection.total_evaluations,
            "total_tokens": projection.total_tokens,
            "total_cost_usd": round(projection.total_cost_usd, 4),
            "within_budget": projection.within_budget,
            "budget_limit": projection.budget_limit,
            "projection_confidence": round(projection.projection_confidence, 3),
            "breakdown_by_model": {
                model: {
                    "prompt_tokens": est.prompt_tokens,
                    "completion_tokens": est.completion_tokens,
                    "total_tokens": est.total_tokens,
                    "cost_usd": round(est.cost_usd, 6),
                    "provider": est.provider
                }
                for model, est in projection.breakdown_by_model.items()
            },
            "breakdown_by_transform": {
                transform: round(cost, 4)
                for transform, cost in projection.breakdown_by_transform.items()
            },
            "breakdown_by_dataset": {
                dataset: round(cost, 4)
                for dataset, cost in projection.breakdown_by_dataset.items()
            },
            "generated_at": logger.handlers[0].formatter.formatTime(logger.makeRecord(
                'cost_estimator', logging.INFO, '', 0, '', (), None
            )) if logger.handlers else "unknown"
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Cost projection exported to {output_path}")


async def create_sample_prompts_from_datasets(
    config: ScrambleBenchConfig, 
    max_samples: int = 50
) -> List[str]:
    """Create sample prompts from configured datasets for cost estimation."""
    sample_prompts = []
    
    for dataset_config in config.datasets:
        try:
            dataset_path = Path(dataset_config.path)
            if not dataset_path.exists():
                logger.warning(f"Dataset not found for cost estimation: {dataset_path}")
                continue
            
            # Load small sample of dataset
            with open(dataset_path, 'r') as f:
                if dataset_path.suffix.lower() == '.jsonl':
                    items = []
                    for i, line in enumerate(f):
                        if i >= max_samples // len(config.datasets):
                            break
                        line = line.strip()
                        if line:
                            try:
                                items.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        items = data[:max_samples // len(config.datasets)]
                    else:
                        items = [data]
            
            # Convert items to prompts
            for item in items:
                if 'question' in item:
                    prompt = f"Q: {item['question']}\nA:"
                elif 'prompt' in item:
                    prompt = item['prompt']
                elif 'text' in item:
                    prompt = item['text']
                else:
                    # Fallback - use item as prompt
                    prompt = str(item)[:500]  # Limit length
                
                # Apply evaluation template
                template = config.evaluation.prompting.template
                if '{question}' in template:
                    formatted_prompt = template.format(question=item.get('question', prompt))
                else:
                    formatted_prompt = prompt
                
                sample_prompts.append(formatted_prompt)
        
        except Exception as e:
            logger.warning(f"Failed to load samples from {dataset_config.path}: {e}")
    
    # Add some default prompts if we couldn't load any
    if not sample_prompts:
        sample_prompts = [
            "Q: What is 2+2?\nA:",
            "Q: Explain the concept of democracy in one sentence.\nA:",
            "Q: Complete the pattern: 1, 2, 4, 8, ?\nA:",
            "Q: What is the capital of France?\nA:",
            "Q: Solve for x: 2x + 5 = 13\nA:"
        ]
        logger.info("Using default sample prompts for cost estimation")
    
    return sample_prompts
