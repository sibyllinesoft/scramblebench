"""
Translation benchmark implementation for ScrambleBench.

This module implements the translation benchmark that evaluates models
on problems translated into constructed languages to avoid training
data contamination.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import time
from pathlib import Path

from scramblebench.core.benchmark import BaseBenchmark, BenchmarkResult
from scramblebench.core.evaluator import Evaluator, EvaluationMode
from scramblebench.translation.language_generator import (
    LanguageGenerator, ConstructedLanguage, LanguageType
)
from scramblebench.translation.translator import ProblemTranslator, TranslatedProblem
from scramblebench.utils.config import Config
from scramblebench.utils.data_loader import DataLoader
from scramblebench.llm.model_adapter import ModelAdapter
from scramblebench.core.data_extractors import ProblemTextExtractor, AnswerExtractor
from scramblebench.core.metrics_computer import TranslationMetricsComputer


class TranslationBenchmark(BaseBenchmark):
    """
    Benchmark that evaluates models on problems translated to constructed languages.
    
    This benchmark loads problems from existing datasets, translates them into
    constructed languages, and evaluates model performance while providing
    translation keys for verification.
    """
    
    def __init__(
        self,
        source_dataset: str,
        language_type: LanguageType = LanguageType.SUBSTITUTION,
        language_complexity: int = 5,
        config: Optional[Config] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the translation benchmark.
        
        Args:
            source_dataset: Name/path of the source dataset to translate
            language_type: Type of constructed language to use
            language_complexity: Complexity level of the language (1-10)
            config: Configuration object
            logger: Logger instance
        """
        super().__init__(
            name=f"translation_{source_dataset}_{language_type.value}",
            config=config,
            logger=logger
        )
        
        self.source_dataset = source_dataset
        self.language_type = language_type
        self.language_complexity = language_complexity
        
        # Initialize components
        self.language_generator = LanguageGenerator(
            seed=self.config.get('random_seed', 42),
            logger=self.logger
        )
        self.translator = ProblemTranslator(logger=self.logger)
        self.evaluator = Evaluator(logger=self.logger)
        self.data_loader = DataLoader(config=self.config, logger=self.logger)
        self.model_adapter = ModelAdapter(logger=self.logger)
        self.problem_extractor = ProblemTextExtractor()
        self.answer_extractor = AnswerExtractor()
        self.metrics_computer = TranslationMetricsComputer()
        
        # Benchmark state
        self.constructed_language: Optional[ConstructedLanguage] = None
        self.translated_problems: List[TranslatedProblem] = []
        self.original_problems: List[Dict[str, Any]] = []
    
    def prepare_data(self) -> None:
        """
        Prepare benchmark data by loading source dataset and generating translations.
        """
        self.logger.info(f"Preparing translation benchmark data for {self.source_dataset}")
        
        # Load original problems
        self.original_problems = self.data_loader.load_dataset(self.source_dataset)
        self.logger.info(f"Loaded {len(self.original_problems)} problems")
        
        # Generate constructed language
        language_name = f"{self.source_dataset}_{self.language_type.value}"
        self.constructed_language = self.language_generator.generate_language(
            name=language_name,
            language_type=self.language_type,
            complexity=self.language_complexity,
            vocab_size=self.config.get('vocab_size', 1000)
        )
        
        # Save language for reproducibility
        language_dir = Path(self.config.get('languages_dir', 'data/languages'))
        language_dir.mkdir(parents=True, exist_ok=True)
        
        language_path = language_dir / f"{language_name}.json"
        self.language_generator.save_language(self.constructed_language, language_path)
        
        # Translate problems
        self.translated_problems = []
        for i, problem in enumerate(self.original_problems):
            self.logger.debug(f"Translating problem {i+1}/{len(self.original_problems)}")
            
            translated = self.translator.translate_problem(
                problem=problem,
                language=self.constructed_language,
                preserve_numbers=self.config.get('preserve_numbers', True),
                preserve_proper_nouns=self.config.get('preserve_proper_nouns', True)
            )
            
            self.translated_problems.append(translated)
        
        self.logger.info(f"Translated {len(self.translated_problems)} problems")
        
        # Verify translations
        self._verify_translations()
    
    def get_evaluation_data(self, num_samples: Optional[int] = None) -> List[TranslatedProblem]:
        """
        Get the translated problems for evaluation.
        
        Args:
            num_samples: Number of samples to return (None for all)
            
        Returns:
            List of translated problems to evaluate
        """
        if num_samples is None:
            return self.translated_problems
        
        return self.translated_problems[:num_samples]
    
    def run_single_evaluation(
        self,
        model: Any,
        data_item: TranslatedProblem
    ) -> Dict[str, Any]:
        """
        Run evaluation on a single translated problem.
        
        Args:
            model: The model to evaluate
            data_item: Translated problem to evaluate
            
        Returns:
            Dictionary containing evaluation results
        """
        # Extract problem components
        problem_text = self.problem_extractor.extract(data_item.translated_problem)
        expected_answer = self.answer_extractor.extract(data_item.original_problem)
        
        # Get model response
        start_time = time.time()
        query_result = self.model_adapter.query(model, problem_text)
        response_time = time.time() - start_time
        
        if not query_result.success:
            self.logger.error(f"Model query failed: {query_result.error}")
            model_response = ""
        else:
            model_response = query_result.text
        
        # Extract answer from response
        extracted_answer = self.evaluator.extract_answer_pattern(model_response)
        if extracted_answer is None:
            extracted_answer = model_response.strip()
        
        # Translate model answer back to original language for evaluation
        translated_answer = self.translator.translate_answer(
            extracted_answer,
            data_item.translation_key,
            reverse=True  # Translate from constructed language back to English
        )
        
        # Evaluate the answer
        evaluation_mode = EvaluationMode(
            self.config.get('evaluation_mode', 'exact_match')
        )
        
        evaluation_result = self.evaluator.evaluate_response(
            predicted=translated_answer,
            expected=expected_answer,
            mode=evaluation_mode,
            threshold=self.config.get('evaluation_threshold', 0.8)
        )
        
        return {
            'problem_id': data_item.original_problem.get('id', 'unknown'),
            'translated_problem': problem_text,
            'original_answer': expected_answer,
            'model_response': model_response,
            'extracted_answer': extracted_answer,
            'translated_answer': translated_answer,
            'correct': evaluation_result.correct,
            'score': evaluation_result.score,
            'evaluation_explanation': evaluation_result.explanation,
            'response_time': response_time,
            'translation_key': data_item.translation_key,
            'language_name': data_item.language_name,
            'metadata': {
                'evaluation_mode': evaluation_mode.value,
                'translation_units': len(data_item.translation_units),
                'language_complexity': self.language_complexity
            }
        }
    
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute aggregate metrics from evaluation results.
        
        Args:
            results: List of individual evaluation results
            
        Returns:
            Dictionary containing computed metrics
        """
        if not results:
            return {'score': 0.0}
        
        # Use the metrics computer for standardized computation
        metrics = self.metrics_computer.compute_basic_accuracy_metrics(results)
        metrics.update(self.metrics_computer.compute_timing_metrics(results))
        metrics.update(self.metrics_computer.compute_difficulty_analysis(results))
        metrics.update(self.metrics_computer.compute_translation_specific_metrics(results))
        metrics.update(self.metrics_computer.compute_confidence_intervals(results))
        
        # Add benchmark-specific information
        metrics.update({
            'language_type': self.language_type.value,
            'language_complexity': self.language_complexity,
            'source_dataset': self.source_dataset
        })
        
        return metrics
    
    
    
    
    def _verify_translations(self) -> None:
        """Verify the quality and consistency of translations."""
        self.logger.info("Verifying translation quality")
        
        total_issues = 0
        for i, translated_problem in enumerate(self.translated_problems):
            verification = self.translator.verify_translation_consistency(
                translated_problem
            )
            
            if not verification['consistent']:
                total_issues += len(verification['issues'])
                self.logger.warning(
                    f"Translation issues in problem {i}: {verification['issues']}"
                )
        
        if total_issues > 0:
            self.logger.warning(f"Found {total_issues} translation issues")
        else:
            self.logger.info("All translations verified successfully")
    
    def export_translation_key(self, output_path: Path) -> None:
        """
        Export the complete translation key for external verification.
        
        Args:
            output_path: Path to save the translation key
        """
        if not self.translated_problems:
            self.logger.warning("No translated problems available for export")
            return
        
        # Aggregate all translation keys
        complete_key = {}
        for problem in self.translated_problems:
            complete_key.update(problem.translation_key)
        
        # Add language metadata
        export_data = {
            'language_name': self.constructed_language.name,
            'language_type': self.language_type.value,
            'language_complexity': self.language_complexity,
            'source_dataset': self.source_dataset,
            'translation_key': complete_key,
            'language_rules': [
                {
                    'source': rule.source,
                    'target': rule.target,
                    'type': rule.rule_type,
                    'priority': rule.priority
                }
                for rule in self.constructed_language.rules
            ],
            'vocabulary': self.constructed_language.vocabulary,
            'statistics': {
                'total_problems': len(self.translated_problems),
                'unique_translations': len(complete_key),
                'total_translation_units': sum(
                    len(p.translation_units) for p in self.translated_problems
                )
            }
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Translation key exported to {output_path}")
    
    def validate_config(self) -> bool:
        """Validate the benchmark configuration."""
        if not super().validate_config():
            return False
        
        # Check required configuration
        required_fields = ['random_seed']
        for field in required_fields:
            if field not in self.config:
                self.logger.error(f"Missing required config field: {field}")
                return False
        
        # Validate language complexity
        if not 1 <= self.language_complexity <= 10:
            self.logger.error(f"Language complexity must be 1-10, got {self.language_complexity}")
            return False
        
        return True