"""
Evaluation runner for orchestrating end-to-end ScrambleBench evaluations.

Coordinates dataset processing, transform application, model evaluation,
and result storage with deterministic reproducibility.
"""

import asyncio
import json
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional, Set
from pathlib import Path
import logging

from .unified_config import ScrambleBenchConfig, DatasetConfig
from .database import ScrambleBenchDatabase
from .adapters import create_adapter, BaseModelAdapter, CompletionResult
from ..transforms import OriginalTransform, ParaphraseTransform, ScrambleTransform


logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Orchestrates end-to-end ScrambleBench evaluations."""
    
    def __init__(self, config: ScrambleBenchConfig, db_path: Optional[Path] = None):
        self.config = config
        self.db = ScrambleBenchDatabase(db_path)
        self.adapters: Dict[str, BaseModelAdapter] = {}
        self.transforms: Dict[str, Any] = {}
        self.run_id: Optional[str] = None
        
        # Initialize transforms
        self._initialize_transforms()
    
    def _initialize_transforms(self):
        """Initialize transform strategy instances."""
        for transform_config in self.config.transforms:
            transform_type = transform_config.type
            
            if transform_type == "original":
                transform = OriginalTransform(
                    config=transform_config.dict(),
                    seed=self.config.run.seed
                )
            elif transform_type == "paraphrase":
                transform = ParaphraseTransform(
                    config=transform_config.dict(),
                    seed=self.config.run.seed
                )
            elif transform_type == "scramble":
                transform = ScrambleTransform(
                    config=transform_config.dict(),
                    seed=self.config.run.seed
                )
            else:
                raise ValueError(f"Unknown transform type: {transform_type}")
            
            self.transforms[transform_type] = transform
    
    async def initialize_adapters(self):
        """Initialize model adapters for all configured providers."""
        for model_config in self.config.run.models:
            provider = model_config.provider
            adapter_config = {
                'model_id': model_config.model_id,
                **model_config.provider_config
            }
            
            try:
                adapter = create_adapter(provider, adapter_config)
                
                # Test adapter connectivity
                test_result = await self._test_adapter(adapter)
                if not test_result['success']:
                    logger.warning(f"Adapter test failed for {model_config.model_id}: {test_result['error']}")
                else:
                    logger.info(f"Successfully initialized adapter for {model_config.model_id}")
                
                self.adapters[model_config.model_id] = adapter
                
            except Exception as e:
                logger.error(f"Failed to initialize adapter for {model_config.model_id}: {e}")
                raise
    
    async def _test_adapter(self, adapter: BaseModelAdapter) -> Dict[str, Any]:
        """Test adapter with a simple completion."""
        try:
            result = await adapter.complete(
                "Test prompt", 
                temperature=0.0, 
                max_tokens=10
            )
            return {
                'success': result.success,
                'error': result.error_message if not result.success else None
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def run_evaluation(self, dry_run: bool = False) -> str:
        """Run complete evaluation pipeline."""
        logger.info("Starting ScrambleBench evaluation")
        
        # Generate run ID
        self.run_id = str(uuid.uuid4())
        
        # Initialize adapters
        await self.initialize_adapters()
        
        # Initialize paraphrase transforms with model adapters
        await self._setup_paraphrase_adapters()
        
        if not dry_run:
            # Create run record
            await self._create_run_record()
        
        # Process each dataset
        total_evaluations = 0
        for dataset_config in self.config.datasets:
            logger.info(f"Processing dataset: {dataset_config.name}")
            
            # Load dataset
            dataset_items = await self._load_dataset(dataset_config)
            logger.info(f"Loaded {len(dataset_items)} items from {dataset_config.name}")
            
            if not dry_run:
                # Store dataset items
                await self._store_dataset_items(dataset_items, dataset_config.name)
            
            # Generate all evaluation tasks
            tasks = []
            for model_config in self.config.run.models:
                for transform_config in self.config.transforms:
                    tasks.append((
                        model_config.model_id,
                        dataset_config,
                        transform_config,
                        dataset_items
                    ))
            
            # Execute evaluations with concurrency control
            semaphore = asyncio.Semaphore(self.config.run.max_concurrency)
            
            evaluation_tasks = [
                self._evaluate_model_transform_dataset(
                    semaphore, model_id, dataset_config, transform_config, 
                    dataset_items, dry_run
                )
                for model_id, dataset_config, transform_config, dataset_items in tasks
            ]
            
            results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
            
            # Count successful evaluations
            for result in results:
                if not isinstance(result, Exception) and result is not None:
                    total_evaluations += result
        
        if not dry_run:
            # Compute aggregate metrics
            await self._compute_aggregates()
            
            # Update run completion status
            self.db.update_run_status(self.run_id, "completed", {
                "total_evaluations": total_evaluations,
                "completed_at": datetime.now(timezone.utc).isoformat()
            })
        
        logger.info(f"Evaluation complete. Run ID: {self.run_id}, Total evaluations: {total_evaluations}")
        return self.run_id
    
    async def _setup_paraphrase_adapters(self):
        """Set model adapters for paraphrase transforms."""
        paraphrase_transform = self.transforms.get('paraphrase')
        if paraphrase_transform:
            # Find paraphrase provider from config
            paraphrase_config = None
            for tc in self.config.transforms:
                if tc.type == "paraphrase":
                    paraphrase_config = tc
                    break
            
            if paraphrase_config and hasattr(paraphrase_config, 'provider'):
                provider_id = paraphrase_config.provider
                if provider_id in self.adapters:
                    paraphrase_transform.set_model_adapter(self.adapters[provider_id])
                    logger.info(f"Set paraphrase adapter to {provider_id}")
                else:
                    logger.warning(f"Paraphrase provider {provider_id} not found in adapters")
    
    async def _create_run_record(self):
        """Create run record in database."""
        run_metadata = {
            "config_hash": self._compute_config_hash(),
            "seed": self.config.run.seed,
            "max_concurrency": self.config.run.max_concurrency,
            "models": [m.model_id for m in self.config.run.models],
            "datasets": [d.name for d in self.config.datasets],
            "transforms": [t.type for t in self.config.transforms],
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        self.db.create_run(
            run_id=self.run_id,
            status="running",
            metadata=run_metadata
        )
        
        logger.info(f"Created run record: {self.run_id}")
    
    def _compute_config_hash(self) -> str:
        """Compute deterministic hash of configuration."""
        config_json = self.config.json(sort_keys=True)
        return hashlib.sha256(config_json.encode()).hexdigest()[:16]
    
    async def _load_dataset(self, dataset_config: DatasetConfig) -> List[Dict[str, Any]]:
        """Load dataset from file."""
        dataset_path = Path(dataset_config.path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            if dataset_path.suffix.lower() == '.jsonl':
                # JSON Lines format
                items = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            item['_line_num'] = line_num
                            items.append(item)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                return items
            else:
                # Regular JSON format
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    # Assume single item
                    return [data]
                else:
                    raise ValueError(f"Unexpected data format in {dataset_path}")
    
    async def _store_dataset_items(self, items: List[Dict[str, Any]], dataset_name: str):
        """Store dataset items in database."""
        for item in items:
            # Generate deterministic item ID
            item_content = json.dumps(item, sort_keys=True)
            item_id = hashlib.sha256(item_content.encode()).hexdigest()[:16]
            
            # Extract text and metadata
            text = item.get('text', item.get('prompt', str(item)))
            
            item_metadata = {
                "original_item": item,
                "dataset": dataset_name
            }
            
            self.db.store_item(
                run_id=self.run_id,
                item_id=item_id,
                dataset=dataset_name,
                domain=item.get('domain', 'general'),
                text=text,
                metadata=item_metadata
            )
    
    async def _evaluate_model_transform_dataset(
        self,
        semaphore: asyncio.Semaphore,
        model_id: str,
        dataset_config: DatasetConfig,
        transform_config: Any,
        dataset_items: List[Dict[str, Any]],
        dry_run: bool = False
    ) -> int:
        """Evaluate single (model, transform, dataset) combination."""
        async with semaphore:
            logger.info(f"Evaluating {model_id} on {dataset_config.name} with {transform_config.type} transform")
            
            adapter = self.adapters[model_id]
            transform = self.transforms[transform_config.type]
            
            evaluations_completed = 0
            
            for item in dataset_items:
                try:
                    # Apply transform to item text
                    original_text = item.get('text', item.get('prompt', str(item)))
                    
                    if transform_config.type == "scramble":
                        # Apply all scramble levels
                        levels = getattr(transform_config, 'levels', [0.1, 0.2, 0.3, 0.4, 0.5])
                        for level in levels:
                            await self._evaluate_single_item(
                                adapter, transform, item, original_text, 
                                dataset_config.name, model_id, transform_config.type,
                                dry_run, scramble_level=level
                            )
                            evaluations_completed += 1
                    else:
                        # Single evaluation for original/paraphrase
                        await self._evaluate_single_item(
                            adapter, transform, item, original_text,
                            dataset_config.name, model_id, transform_config.type,
                            dry_run
                        )
                        evaluations_completed += 1
                
                except Exception as e:
                    logger.error(f"Failed to evaluate item {item.get('_line_num', '?')}: {e}")
                    continue
            
            logger.info(f"Completed {evaluations_completed} evaluations for {model_id}/{transform_config.type}")
            return evaluations_completed
    
    async def _evaluate_single_item(
        self,
        adapter: BaseModelAdapter,
        transform: Any,
        item: Dict[str, Any],
        original_text: str,
        dataset_name: str,
        model_id: str,
        transform_type: str,
        dry_run: bool = False,
        **transform_kwargs
    ):
        """Evaluate single item with transform."""
        # Generate item ID
        item_content = json.dumps(item, sort_keys=True)
        item_id = hashlib.sha256(item_content.encode()).hexdigest()[:16]
        
        # Apply transform
        if transform_type == "scramble":
            transform_result = transform.transform(original_text, **transform_kwargs)
        elif hasattr(transform, 'transform_async'):
            transform_result = await transform.transform_async(original_text, item_id=item_id)
        else:
            transform_result = transform.transform(original_text, item_id=item_id)
        
        if not transform_result.success:
            logger.warning(f"Transform failed for item {item_id}: {transform_result.error_message}")
            return
        
        transformed_text = transform_result.transformed_text
        
        # Create prompt from transformed text
        prompt = self._create_evaluation_prompt(transformed_text, item)
        
        if dry_run:
            logger.info(f"[DRY RUN] Would evaluate: {model_id} on item {item_id[:8]}...")
            return
        
        # Get model completion
        completion = await adapter.complete(
            prompt=prompt,
            temperature=0.0,  # Deterministic evaluation
            max_tokens=self.config.run.max_tokens,
            seed=self.config.run.seed
        )
        
        if not completion.success:
            logger.warning(f"Completion failed for item {item_id}: {completion.error_message}")
            return
        
        # Extract answer and compute accuracy
        predicted_answer = self._extract_answer(completion.text, item)
        correct_answer = item.get('answer', item.get('label'))
        
        accuracy = self._compute_accuracy(predicted_answer, correct_answer, item)
        
        # Store evaluation result
        eval_metadata = {
            "transform_metadata": transform_result.transform_metadata,
            "completion_metadata": completion.metadata,
            "prompt": prompt,
            "predicted_answer": predicted_answer,
            "correct_answer": correct_answer,
            "transform_kwargs": transform_kwargs
        }
        
        self.db.store_evaluation(
            run_id=self.run_id,
            item_id=item_id,
            model_id=model_id,
            transform_type=transform_type,
            accuracy=accuracy,
            response_text=completion.text,
            tokens_used=completion.tokens_used or 0,
            cost_usd=completion.cost_usd or 0.0,
            metadata=eval_metadata
        )
    
    def _create_evaluation_prompt(self, transformed_text: str, item: Dict[str, Any]) -> str:
        """Create evaluation prompt from transformed text and item."""
        # Simple prompt format - can be made more sophisticated
        if 'question' in item:
            return f"{transformed_text}\n\nQuestion: {item['question']}\nAnswer:"
        elif 'prompt' in item:
            return transformed_text
        else:
            # Fallback: assume text is the prompt
            return transformed_text
    
    def _extract_answer(self, response_text: str, item: Dict[str, Any]) -> str:
        """Extract predicted answer from model response."""
        # Simple extraction - take first non-empty line
        lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]
        return lines[0] if lines else ""
    
    def _compute_accuracy(self, predicted: str, correct: Any, item: Dict[str, Any]) -> float:
        """Compute accuracy score for prediction."""
        if correct is None:
            return 0.0
        
        # Normalize both answers
        pred_normalized = str(predicted).strip().lower()
        correct_normalized = str(correct).strip().lower()
        
        # Exact match
        if pred_normalized == correct_normalized:
            return 1.0
        
        # Check if prediction contains correct answer
        if correct_normalized in pred_normalized:
            return 1.0
        
        # For multiple choice, extract letter/number
        pred_choice = self._extract_choice(pred_normalized)
        correct_choice = self._extract_choice(correct_normalized)
        
        if pred_choice and correct_choice and pred_choice == correct_choice:
            return 1.0
        
        return 0.0
    
    def _extract_choice(self, text: str) -> Optional[str]:
        """Extract choice letter/number from text."""
        import re
        
        # Look for patterns like "A)", "1.", "(B)", etc.
        patterns = [
            r'\b([A-E])\)',
            r'\(([A-E])\)',
            r'\b([A-E])\.',
            r'\b([1-5])\)',
            r'\(([1-5])\)',
            r'\b([1-5])\.'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).upper()
        
        return None
    
    async def _compute_aggregates(self):
        """Compute aggregate metrics and store in database."""
        logger.info("Computing aggregate metrics")
        
        # Get all unique combinations for aggregation
        combinations = self.db.get_eval_combinations(self.run_id)
        
        for model_id, dataset, domain in combinations:
            # Compute canonical metrics (RRS, LDC)
            metrics = self.db.compute_canonical_metrics(self.run_id, model_id, dataset, domain)
            
            if metrics:
                # Store aggregate results
                self.db.store_aggregate(
                    run_id=self.run_id,
                    model_id=model_id,
                    dataset=dataset,
                    domain=domain,
                    metric_type="canonical",
                    metric_value=metrics.get('rrs', 0.0),
                    metadata={
                        "rrs": metrics.get('rrs', 0.0),
                        "ldc": metrics.get('ldc', 0.0),
                        "acc_original": metrics.get('acc_original', 0.0),
                        "acc_scrambled": metrics.get('acc_scrambled', 0.0),
                        "n_items": metrics.get('n_items', 0)
                    }
                )
        
        logger.info("Aggregate computation complete")
    
    async def get_run_results(self, run_id: str) -> Dict[str, Any]:
        """Get comprehensive results for a run."""
        # Get run info
        run_info = self.db.get_run_info(run_id)
        if not run_info:
            raise ValueError(f"Run not found: {run_id}")
        
        # Get aggregate results
        aggregates = self.db.get_aggregates(run_id)
        
        # Get evaluation counts
        eval_counts = self.db.get_evaluation_counts(run_id)
        
        return {
            "run_info": run_info,
            "aggregates": aggregates,
            "evaluation_counts": eval_counts,
            "summary": {
                "total_evaluations": sum(eval_counts.values()),
                "models": list(set(agg['model_id'] for agg in aggregates)),
                "datasets": list(set(agg['dataset'] for agg in aggregates))
            }
        }
    
    async def export_results(self, run_id: str, output_path: Path, format: str = "json") -> Path:
        """Export results to file."""
        results = await self.get_run_results(run_id)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Results exported to {output_path}")
        return output_path