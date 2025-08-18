"""
Transformation pipeline for generating transformed versions of benchmark questions.
"""

import asyncio
import logging
import random
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json

from tqdm.asyncio import tqdm as atqdm

from scramblebench.translation.language_generator import LanguageGenerator, LanguageType
from scramblebench.translation.translator import ProblemTranslator
from scramblebench.translation.text_transformer import (
    ProperNounSwapper, SynonymReplacer, VocabularyExtractor
)
from scramblebench.utils.data_loader import DataLoader
from .config import TransformationConfig, TransformationType


@dataclass
class TransformationResult:
    """Result of applying a transformation to a problem."""
    original_problem: Dict[str, Any]
    transformed_problem: Dict[str, Any]
    transformation_type: str
    transformation_metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None


@dataclass
class TransformationSet:
    """Collection of transformations for a single problem."""
    original_problem: Dict[str, Any]
    transformations: List[TransformationResult]
    problem_id: str
    
    def get_successful_transformations(self) -> List[TransformationResult]:
        """Get only successful transformations."""
        return [t for t in self.transformations if t.success]
    
    def get_by_type(self, transformation_type: str) -> List[TransformationResult]:
        """Get transformations of a specific type."""
        return [t for t in self.transformations if t.transformation_type == transformation_type]


class TransformationPipeline:
    """
    Pipeline for generating transformed versions of benchmark questions.
    
    Supports multiple transformation types including language translation,
    proper noun swapping, synonym replacement, and paraphrasing.
    """
    
    def __init__(
        self,
        config: TransformationConfig,
        data_dir: Optional[Path] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the transformation pipeline.
        
        Args:
            config: Transformation configuration
            data_dir: Directory for storing languages and temporary files
            logger: Logger instance
        """
        self.config = config
        self.data_dir = data_dir or Path("data")
        self.languages_dir = self.data_dir / "languages"
        self.languages_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        self.language_generator = LanguageGenerator(seed=config.seed)
        self.problem_translator = ProblemTranslator()
        self.data_loader = DataLoader()
        
        # Generated languages cache
        self._generated_languages: Dict[str, Any] = {}
        
        # Set random seed if provided
        if config.seed is not None:
            random.seed(config.seed)
    
    async def generate_transformation_sets(
        self,
        problems: List[Dict[str, Any]],
        problem_ids: Optional[List[str]] = None
    ) -> List[TransformationSet]:
        """
        Generate transformation sets for a list of problems.
        
        Args:
            problems: List of benchmark problems
            problem_ids: Optional list of problem IDs (auto-generated if None)
            
        Returns:
            List of transformation sets
        """
        if problem_ids is None:
            problem_ids = [f"problem_{i}" for i in range(len(problems))]
        
        if len(problem_ids) != len(problems):
            raise ValueError("Number of problem_ids must match number of problems")
        
        self.logger.info(f"Generating transformations for {len(problems)} problems")
        
        # Prepare languages if needed
        await self._prepare_languages()
        
        # Generate transformations for each problem
        transformation_sets = []
        
        # Use ThreadPoolExecutor for CPU-bound transformation tasks
        with ThreadPoolExecutor(max_workers=4) as executor:
            tasks = []
            for problem, problem_id in zip(problems, problem_ids):
                task = asyncio.get_event_loop().run_in_executor(
                    executor, self._generate_transformations_for_problem, problem, problem_id
                )
                tasks.append(task)
            
            # Execute with progress bar
            for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating transformations"):
                transformation_set = await coro
                transformation_sets.append(transformation_set)
        
        self.logger.info(f"Generated {len(transformation_sets)} transformation sets")
        return transformation_sets
    
    def _generate_transformations_for_problem(
        self,
        problem: Dict[str, Any],
        problem_id: str
    ) -> TransformationSet:
        """Generate all transformations for a single problem."""
        transformations = []
        
        for transform_type in self.config.enabled_types:
            if transform_type == TransformationType.ALL:
                # Generate all transformation types
                for t_type in TransformationType:
                    if t_type != TransformationType.ALL:
                        result = self._apply_transformation(problem, t_type)
                        if result:
                            transformations.append(result)
            else:
                result = self._apply_transformation(problem, transform_type)
                if result:
                    transformations.append(result)
        
        return TransformationSet(
            original_problem=problem,
            transformations=transformations,
            problem_id=problem_id
        )
    
    def _apply_transformation(
        self,
        problem: Dict[str, Any],
        transformation_type: TransformationType
    ) -> Optional[TransformationResult]:
        """Apply a specific transformation to a problem."""
        try:
            if transformation_type == TransformationType.LANGUAGE_TRANSLATION:
                return self._apply_language_translation(problem)
            elif transformation_type == TransformationType.PROPER_NOUN_SWAP:
                return self._apply_proper_noun_swap(problem)
            elif transformation_type == TransformationType.SYNONYM_REPLACEMENT:
                return self._apply_synonym_replacement(problem)
            elif transformation_type == TransformationType.PARAPHRASING:
                return self._apply_paraphrasing(problem)
            elif transformation_type == TransformationType.LONG_CONTEXT:
                return self._apply_long_context(problem)
            else:
                self.logger.warning(f"Unknown transformation type: {transformation_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error applying {transformation_type} transformation: {e}")
            return TransformationResult(
                original_problem=problem,
                transformed_problem=problem,
                transformation_type=transformation_type.value,
                transformation_metadata={"error": str(e)},
                success=False,
                error=str(e)
            )
    
    def _apply_language_translation(self, problem: Dict[str, Any]) -> TransformationResult:
        """Apply constructed language translation."""
        # Select a random language from configured languages
        language_name = random.choice(self.config.languages)
        
        if language_name not in self._generated_languages:
            raise ValueError(f"Language {language_name} not generated")
        
        language = self._generated_languages[language_name]
        
        # Translate the problem
        translated = self.problem_translator.translate_problem(
            problem,
            language,
            preserve_numbers=True,
            preserve_proper_nouns=True
        )
        
        return TransformationResult(
            original_problem=problem,
            transformed_problem=translated.translated_problem,
            transformation_type=TransformationType.LANGUAGE_TRANSLATION.value,
            transformation_metadata={
                "language": language_name,
                "translation_key": translated.translation_key,
                "preserve_numbers": True,
                "preserve_proper_nouns": True
            },
            success=True
        )
    
    def _apply_proper_noun_swap(self, problem: Dict[str, Any]) -> TransformationResult:
        """Apply proper noun swapping."""
        swapper = ProperNounSwapper(
            strategy=self.config.proper_noun_strategy,
            seed=self.config.seed
        )
        
        # Apply to all text fields in the problem
        transformed_problem = problem.copy()
        all_replacements = {}
        
        for key, value in problem.items():
            if isinstance(value, str):
                result = swapper.transform_text(value)
                transformed_problem[key] = result.transformed_text
                all_replacements.update(result.replacements)
        
        return TransformationResult(
            original_problem=problem,
            transformed_problem=transformed_problem,
            transformation_type=TransformationType.PROPER_NOUN_SWAP.value,
            transformation_metadata={
                "strategy": self.config.proper_noun_strategy,
                "replacements": all_replacements
            },
            success=True
        )
    
    def _apply_synonym_replacement(self, problem: Dict[str, Any]) -> TransformationResult:
        """Apply synonym replacement."""
        replacer = SynonymReplacer(
            replacement_rate=self.config.synonym_rate,
            preserve_function_words=self.config.preserve_function_words,
            seed=self.config.seed
        )
        
        # Apply to all text fields in the problem
        transformed_problem = problem.copy()
        all_replacements = {}
        
        for key, value in problem.items():
            if isinstance(value, str):
                result = replacer.transform_text(value)
                transformed_problem[key] = result.transformed_text
                all_replacements.update(result.replacements)
        
        return TransformationResult(
            original_problem=problem,
            transformed_problem=transformed_problem,
            transformation_type=TransformationType.SYNONYM_REPLACEMENT.value,
            transformation_metadata={
                "replacement_rate": self.config.synonym_rate,
                "preserve_function_words": self.config.preserve_function_words,
                "replacements": all_replacements
            },
            success=True
        )
    
    def _apply_paraphrasing(self, problem: Dict[str, Any]) -> TransformationResult:
        """Apply paraphrasing transformation (placeholder for now)."""
        # This would require an LLM-based paraphrasing system
        # For now, return a copy of the original problem
        self.logger.warning("Paraphrasing transformation not yet implemented")
        
        return TransformationResult(
            original_problem=problem,
            transformed_problem=problem.copy(),
            transformation_type=TransformationType.PARAPHRASING.value,
            transformation_metadata={"status": "not_implemented"},
            success=False,
            error="Paraphrasing transformation not yet implemented"
        )
    
    def _apply_long_context(self, problem: Dict[str, Any]) -> TransformationResult:
        """Apply long context transformation (placeholder for now)."""
        # This would use the longcontext module
        # For now, return a copy of the original problem
        self.logger.warning("Long context transformation not yet implemented")
        
        return TransformationResult(
            original_problem=problem,
            transformed_problem=problem.copy(),
            transformation_type=TransformationType.LONG_CONTEXT.value,
            transformation_metadata={"status": "not_implemented"},
            success=False,
            error="Long context transformation not yet implemented"
        )
    
    async def _prepare_languages(self) -> None:
        """Prepare constructed languages for translation."""
        self.logger.info("Preparing constructed languages...")
        
        for language_name in self.config.languages:
            if language_name in self._generated_languages:
                continue
            
            # Check if language file already exists
            lang_file = self.languages_dir / f"{language_name}.json"
            
            if lang_file.exists():
                self.logger.info(f"Loading existing language: {language_name}")
                language = self.language_generator.load_language(lang_file)
            else:
                self.logger.info(f"Generating new language: {language_name}")
                
                # Generate language based on name pattern
                if "agglutinative" in language_name.lower():
                    lang_type = LanguageType.AGGLUTINATIVE
                elif "fusional" in language_name.lower():
                    lang_type = LanguageType.FUSIONAL
                elif "isolating" in language_name.lower():
                    lang_type = LanguageType.ISOLATING
                else:
                    # Default to agglutinative
                    lang_type = LanguageType.AGGLUTINATIVE
                
                language = self.language_generator.generate_language(
                    name=language_name,
                    language_type=lang_type,
                    complexity=self.config.language_complexity,
                    vocab_size=1000
                )
                
                # Save for future use
                self.language_generator.save_language(language, lang_file)
            
            self._generated_languages[language_name] = language
    
    def save_transformation_sets(
        self,
        transformation_sets: List[TransformationSet],
        output_path: Path
    ) -> None:
        """Save transformation sets to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        data = {
            "config": self.config.dict(),
            "transformation_sets": []
        }
        
        for ts in transformation_sets:
            ts_data = {
                "problem_id": ts.problem_id,
                "original_problem": ts.original_problem,
                "transformations": []
            }
            
            for t in ts.transformations:
                t_data = {
                    "transformation_type": t.transformation_type,
                    "transformed_problem": t.transformed_problem,
                    "transformation_metadata": t.transformation_metadata,
                    "success": t.success,
                    "error": t.error
                }
                ts_data["transformations"].append(t_data)
            
            data["transformation_sets"].append(ts_data)
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved {len(transformation_sets)} transformation sets to {output_path}")
    
    @classmethod
    def load_transformation_sets(cls, input_path: Path) -> Tuple[List[TransformationSet], TransformationConfig]:
        """Load transformation sets from file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        config = TransformationConfig(**data["config"])
        transformation_sets = []
        
        for ts_data in data["transformation_sets"]:
            transformations = []
            for t_data in ts_data["transformations"]:
                transformation = TransformationResult(
                    original_problem=ts_data["original_problem"],
                    transformed_problem=t_data["transformed_problem"],
                    transformation_type=t_data["transformation_type"],
                    transformation_metadata=t_data["transformation_metadata"],
                    success=t_data["success"],
                    error=t_data.get("error")
                )
                transformations.append(transformation)
            
            ts = TransformationSet(
                original_problem=ts_data["original_problem"],
                transformations=transformations,
                problem_id=ts_data["problem_id"]
            )
            transformation_sets.append(ts)
        
        return transformation_sets, config
    
    def get_transformation_stats(self, transformation_sets: List[TransformationSet]) -> Dict[str, Any]:
        """Get statistics about transformations."""
        total_problems = len(transformation_sets)
        total_transformations = sum(len(ts.transformations) for ts in transformation_sets)
        successful_transformations = sum(len(ts.get_successful_transformations()) for ts in transformation_sets)
        
        # Count by type
        type_counts = {}
        type_success_counts = {}
        
        for ts in transformation_sets:
            for t in ts.transformations:
                t_type = t.transformation_type
                type_counts[t_type] = type_counts.get(t_type, 0) + 1
                if t.success:
                    type_success_counts[t_type] = type_success_counts.get(t_type, 0) + 1
        
        # Calculate success rates
        type_success_rates = {
            t_type: type_success_counts.get(t_type, 0) / count
            for t_type, count in type_counts.items()
        }
        
        return {
            "total_problems": total_problems,
            "total_transformations": total_transformations,
            "successful_transformations": successful_transformations,
            "overall_success_rate": successful_transformations / total_transformations if total_transformations > 0 else 0,
            "transformations_by_type": type_counts,
            "success_rates_by_type": type_success_rates
        }