"""
Long context benchmark implementation for ScrambleBench.

This module implements benchmarks for evaluating models on long documents
and Q&A sets through various transformation strategies to avoid training
data contamination while maintaining evaluation validity.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import time
from pathlib import Path
import json
import re

from scramblebench.core.benchmark import BaseBenchmark, BenchmarkResult
from scramblebench.core.evaluator import Evaluator, EvaluationMode
from scramblebench.longcontext.document_transformer import (
    DocumentTransformer, TransformedDocument, TransformationType
)
from scramblebench.longcontext.qa_transformer import (
    QATransformer, QAPair, TransformedQAPair, AnswerType
)
from scramblebench.translation.language_generator import (
    LanguageGenerator, ConstructedLanguage, LanguageType
)
from scramblebench.translation.translator import ProblemTranslator
from ..core.unified_config import ScrambleBenchConfig
from scramblebench.utils.data_loader import DataLoader
from scramblebench.llm.model_adapter import ModelAdapter
from scramblebench.core.data_extractors import DocumentExtractor, QAPairExtractor, PromptConstructor
from scramblebench.core.metrics_computer import LongContextMetricsComputer


class LongContextBenchmark(BaseBenchmark):
    """
    Benchmark for evaluating models on long context understanding tasks.
    
    This benchmark transforms long documents and their associated Q&A pairs
    to create contamination-free evaluation while preserving the essential
    information content and evaluation validity.
    """
    
    def __init__(
        self,
        dataset_name: str,
        transformation_type: TransformationType = TransformationType.TRANSLATION,
        language_type: Optional[LanguageType] = LanguageType.SUBSTITUTION,
        language_complexity: int = 5,
        config: Optional[ScrambleBenchConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the long context benchmark.
        
        Args:
            dataset_name: Name/path of the dataset to use
            transformation_type: Type of document transformation
            language_type: Type of constructed language (for translation)
            language_complexity: Complexity level of the language (1-10)
            config: Configuration object
            logger: Logger instance
        """
        super().__init__(
            name=f"longcontext_{dataset_name}_{transformation_type.value}",
            config=config,
            logger=logger
        )
        
        self.dataset_name = dataset_name
        self.transformation_type = transformation_type
        self.language_type = language_type
        self.language_complexity = language_complexity
        
        # Initialize components
        self.language_generator = LanguageGenerator(
            seed=self.config.get('random_seed', 42),
            logger=self.logger
        ) if language_type else None
        
        self.translator = ProblemTranslator(logger=self.logger)
        self.document_transformer = DocumentTransformer(
            translator=self.translator,
            logger=self.logger
        )
        self.qa_transformer = QATransformer(
            translator=self.translator,
            logger=self.logger
        )
        self.evaluator = Evaluator(logger=self.logger)
        self.data_loader = DataLoader(config=self.config, logger=self.logger)
        self.model_adapter = ModelAdapter(logger=self.logger)
        self.document_extractor = DocumentExtractor()
        self.qa_extractor = QAPairExtractor()
        self.prompt_constructor = PromptConstructor(
            self.config.get('prompt_template', "Context: {context}\n\nQuestion: {question}\n\nAnswer:")
        )
        self.metrics_computer = LongContextMetricsComputer()
        
        # Benchmark state
        self.constructed_language: Optional[ConstructedLanguage] = None
        self.original_data: List[Dict[str, Any]] = []
        self.transformed_documents: List[TransformedDocument] = []
        self.transformed_qa_pairs: List[List[TransformedQAPair]] = []
    
    def prepare_data(self) -> None:
        """
        Prepare benchmark data by loading documents and Q&A pairs, then transforming them.
        """
        self.logger.info(f"Preparing long context benchmark data for {self.dataset_name}")
        
        # Load original dataset
        self.original_data = self.data_loader.load_dataset(self.dataset_name)
        self.logger.info(f"Loaded {len(self.original_data)} documents with Q&A pairs")
        
        # Generate constructed language if needed
        if self.transformation_type == TransformationType.TRANSLATION and self.language_type:
            language_name = f"{self.dataset_name}_{self.language_type.value}_longcontext"
            self.constructed_language = self.language_generator.generate_language(
                name=language_name,
                language_type=self.language_type,
                complexity=self.language_complexity,
                vocab_size=self.config.get('vocab_size', 2000)  # Larger vocab for long documents
            )
            
            # Save language
            language_dir = Path(self.config.get('languages_dir', 'data/languages'))
            language_dir.mkdir(parents=True, exist_ok=True)
            
            language_path = language_dir / f"{language_name}.json"
            self.language_generator.save_language(self.constructed_language, language_path)
        
        # Transform documents and Q&A pairs
        self.transformed_documents = []
        self.transformed_qa_pairs = []
        
        for i, data_item in enumerate(self.original_data):
            self.logger.debug(f"Transforming document {i+1}/{len(self.original_data)}")
            
            # Extract document and Q&A pairs
            document = self.document_extractor.extract(data_item)
            qa_pairs_data = self.qa_extractor.extract(data_item)
            
            # Convert to QAPair objects for compatibility
            qa_pairs = []
            for qa_data in qa_pairs_data:
                qa_pair = QAPair(
                    question=qa_data['question'],
                    answer=qa_data['answer'],
                    answer_type=qa_data['answer_type'],
                    context_required=qa_data['context_required'],
                    difficulty=qa_data['difficulty'],
                    metadata=qa_data['metadata']
                )
                qa_pairs.append(qa_pair)
            
            # Transform document
            transformed_doc = self.document_transformer.transform_document(
                document=document,
                transformation_type=self.transformation_type,
                language=self.constructed_language,
                preserve_structure=self.config.get('preserve_structure', True),
                preserve_entities=self.config.get('preserve_entities', True)
            )
            
            # Transform Q&A pairs
            transformed_qa = self.qa_transformer.transform_qa_pairs(
                qa_pairs=qa_pairs,
                transformed_document=transformed_doc,
                language=self.constructed_language
            )
            
            self.transformed_documents.append(transformed_doc)
            self.transformed_qa_pairs.append(transformed_qa)
        
        self.logger.info(
            f"Transformed {len(self.transformed_documents)} documents and "
            f"{sum(len(qa_list) for qa_list in self.transformed_qa_pairs)} Q&A pairs"
        )
        
        # Validate transformations
        self._validate_transformations()
    
    def get_evaluation_data(self, num_samples: Optional[int] = None) -> List[Tuple[TransformedDocument, List[TransformedQAPair]]]:
        """
        Get the transformed documents and Q&A pairs for evaluation.
        
        Args:
            num_samples: Number of documents to return (None for all)
            
        Returns:
            List of (document, qa_pairs) tuples
        """
        data_pairs = list(zip(self.transformed_documents, self.transformed_qa_pairs))
        
        if num_samples is None:
            return data_pairs
        
        return data_pairs[:num_samples]
    
    def run_single_evaluation(
        self,
        model: Any,
        data_item: Tuple[TransformedDocument, List[TransformedQAPair]]
    ) -> Dict[str, Any]:
        """
        Run evaluation on a single document and its Q&A pairs.
        
        Args:
            model: The model to evaluate
            data_item: Tuple of (transformed_document, transformed_qa_pairs)
            
        Returns:
            Dictionary containing evaluation results
        """
        transformed_doc, transformed_qa_pairs = data_item
        
        # Prepare context for the model
        context = transformed_doc.transformed_document
        
        qa_results = []
        total_correct = 0
        total_score = 0.0
        total_response_time = 0.0
        
        for qa_idx, transformed_qa in enumerate(transformed_qa_pairs):
            self.logger.debug(f"Evaluating Q&A {qa_idx+1}/{len(transformed_qa_pairs)}")
            
            # Construct prompt for the model
            prompt = self.prompt_constructor.construct_qa_prompt(context, transformed_qa.transformed_qa.question)
            
            # Get model response
            start_time = time.time()
            query_result = self.model_adapter.query(model, prompt)
            response_time = time.time() - start_time
            total_response_time += response_time
            
            if not query_result.success:
                self.logger.error(f"Model query failed: {query_result.error}")
                model_response = ""
            else:
                model_response = query_result.text
            
            # Extract answer from response
            extracted_answer = self.evaluator.extract_answer_pattern(model_response)
            if extracted_answer is None:
                extracted_answer = model_response.strip()
            
            # Evaluate the answer
            evaluation_mode = EvaluationMode(
                self.config.get('evaluation_mode', 'exact_match')
            )
            
            # For long context, we may need to translate back to original language
            if self.constructed_language:
                # Try to translate model answer back for comparison
                try:
                    translated_back = self.translator.translate_answer(
                        extracted_answer,
                        transformed_qa.answer_mapping,
                        reverse=True
                    )
                    comparison_answer = translated_back
                except:
                    # Fallback to direct comparison
                    comparison_answer = extracted_answer
            else:
                comparison_answer = extracted_answer
            
            evaluation_result = self.evaluator.evaluate_response(
                predicted=comparison_answer,
                expected=transformed_qa.original_qa.answer,
                mode=evaluation_mode,
                threshold=self.config.get('evaluation_threshold', 0.8)
            )
            
            if evaluation_result.correct:
                total_correct += 1
            total_score += evaluation_result.score
            
            qa_result = {
                'qa_index': qa_idx,
                'question': transformed_qa.transformed_qa.question,
                'expected_answer': transformed_qa.transformed_qa.answer,
                'original_answer': transformed_qa.original_qa.answer,
                'model_response': model_response,
                'extracted_answer': extracted_answer,
                'comparison_answer': comparison_answer,
                'correct': evaluation_result.correct,
                'score': evaluation_result.score,
                'evaluation_explanation': evaluation_result.explanation,
                'response_time': response_time,
                'answer_type': transformed_qa.transformed_qa.answer_type,
                'transformation_confidence': transformed_qa.confidence,
                'metadata': {
                    'difficulty': transformed_qa.transformed_qa.difficulty,
                    'context_required': transformed_qa.transformed_qa.context_required
                }
            }
            
            qa_results.append(qa_result)
        
        # Compute document-level metrics
        accuracy = total_correct / len(transformed_qa_pairs) if transformed_qa_pairs else 0.0
        avg_score = total_score / len(transformed_qa_pairs) if transformed_qa_pairs else 0.0
        avg_response_time = total_response_time / len(transformed_qa_pairs) if transformed_qa_pairs else 0.0
        
        return {
            'document_id': transformed_doc.metadata.get('document_id', 'unknown'),
            'document_length': len(transformed_doc.transformed_document),
            'original_document_length': len(transformed_doc.original_document),
            'transformation_type': transformed_doc.transformation_type.value,
            'qa_count': len(transformed_qa_pairs),
            'accuracy': accuracy,
            'avg_score': avg_score,
            'avg_response_time': avg_response_time,
            'total_response_time': total_response_time,
            'qa_results': qa_results,
            'document_stats': self.document_transformer.get_transformation_stats(transformed_doc),
            'qa_stats': self.qa_transformer.get_transformation_stats(transformed_qa_pairs),
            'metadata': {
                'language_name': self.constructed_language.name if self.constructed_language else None,
                'language_complexity': self.language_complexity,
                'transformation_mappings': len(transformed_doc.transformation_map)
            }
        }
    
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute aggregate metrics from evaluation results.
        
        Args:
            results: List of document evaluation results
            
        Returns:
            Dictionary containing computed metrics
        """
        if not results:
            return {'score': 0.0}
        
        # Extract Q&A results for detailed analysis
        all_qa_results = []
        for result in results:
            all_qa_results.extend(result.get('qa_results', []))
        
        if not all_qa_results:
            return {'score': 0.0}
        
        # Use the metrics computer for standardized computation
        metrics = self.metrics_computer.compute_basic_accuracy_metrics(all_qa_results)
        metrics.update(self.metrics_computer.compute_timing_metrics(all_qa_results))
        metrics.update(self.metrics_computer.compute_difficulty_analysis(all_qa_results))
        metrics.update(self.metrics_computer.compute_answer_type_analysis(all_qa_results))
        metrics.update(self.metrics_computer.compute_long_context_specific_metrics(results))
        metrics.update(self.metrics_computer.compute_confidence_intervals(all_qa_results))
        
        # Add benchmark-specific information
        metrics.update({
            'transformation_type': self.transformation_type.value,
            'language_type': self.language_type.value if self.language_type else None,
            'language_complexity': self.language_complexity,
            'dataset_name': self.dataset_name,
            'total_documents': len(results),
            'total_qa_pairs': len(all_qa_results)
        })
        
        # Add transformation quality analysis if available
        transformation_confidences = [
            qa.get('transformation_confidence', 0.0) for qa in all_qa_results
            if 'transformation_confidence' in qa
        ]
        if transformation_confidences:
            metrics['avg_transformation_confidence'] = sum(transformation_confidences) / len(transformation_confidences)
        
        return metrics
    
    
    
    
    
    
    
    def _validate_transformations(self) -> None:
        """Validate the quality of document and Q&A transformations."""
        self.logger.info("Validating transformations")
        
        total_issues = 0
        
        for i, (doc, qa_pairs) in enumerate(zip(self.transformed_documents, self.transformed_qa_pairs)):
            # Validate document transformation
            doc_stats = self.document_transformer.get_transformation_stats(doc)
            
            if doc_stats['length_ratio'] < 0.5 or doc_stats['length_ratio'] > 2.0:
                self.logger.warning(
                    f"Document {i} has unusual length ratio: {doc_stats['length_ratio']}"
                )
                total_issues += 1
            
            # Validate Q&A transformations
            for j, qa_pair in enumerate(qa_pairs):
                validation = self.qa_transformer.validate_transformation(qa_pair, doc)
                
                if not validation['valid']:
                    self.logger.warning(
                        f"Q&A pair {i}.{j} validation failed: {validation['issues']}"
                    )
                    total_issues += len(validation['issues'])
        
        if total_issues > 0:
            self.logger.warning(f"Found {total_issues} transformation issues")
        else:
            self.logger.info("All transformations validated successfully")
    
    def export_benchmark_data(self, output_dir: Path) -> None:
        """
        Export the complete benchmark data for external use.
        
        Args:
            output_dir: Directory to save benchmark data
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export transformed documents
        documents_file = output_dir / "transformed_documents.json"
        docs_data = []
        
        for i, doc in enumerate(self.transformed_documents):
            docs_data.append({
                'id': i,
                'original_document': doc.original_document,
                'transformed_document': doc.transformed_document,
                'transformation_type': doc.transformation_type.value,
                'transformation_map': doc.transformation_map,
                'metadata': doc.metadata,
                'stats': self.document_transformer.get_transformation_stats(doc)
            })
        
        with open(documents_file, 'w') as f:
            json.dump(docs_data, f, indent=2, ensure_ascii=False)
        
        # Export transformed Q&A pairs
        qa_file = output_dir / "transformed_qa_pairs.json"
        qa_data = []
        
        for i, qa_list in enumerate(self.transformed_qa_pairs):
            doc_qa = {
                'document_id': i,
                'qa_pairs': []
            }
            
            for j, qa_pair in enumerate(qa_list):
                doc_qa['qa_pairs'].append({
                    'id': j,
                    'original_question': qa_pair.original_qa.question,
                    'original_answer': qa_pair.original_qa.answer,
                    'transformed_question': qa_pair.transformed_qa.question,
                    'transformed_answer': qa_pair.transformed_qa.answer,
                    'answer_type': qa_pair.transformed_qa.answer_type,
                    'confidence': qa_pair.confidence,
                    'answer_mapping': qa_pair.answer_mapping,
                    'metadata': qa_pair.metadata
                })
            
            qa_data.append(doc_qa)
        
        with open(qa_file, 'w') as f:
            json.dump(qa_data, f, indent=2, ensure_ascii=False)
        
        # Export language if used
        if self.constructed_language:
            language_file = output_dir / "constructed_language.json"
            self.language_generator.save_language(self.constructed_language, language_file)
        
        # Export benchmark metadata
        metadata_file = output_dir / "benchmark_metadata.json"
        metadata = {
            'benchmark_name': self.name,
            'dataset_name': self.dataset_name,
            'transformation_type': self.transformation_type.value,
            'language_type': self.language_type.value if self.language_type else None,
            'language_complexity': self.language_complexity,
            'total_documents': len(self.transformed_documents),
            'total_qa_pairs': sum(len(qa_list) for qa_list in self.transformed_qa_pairs),
            'config': self.config.to_dict()
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Benchmark data exported to {output_dir}")
    
    def validate_config(self) -> bool:
        """Validate the benchmark configuration."""
        if not super().validate_config():
            return False
        
        # Check transformation type compatibility
        if self.transformation_type == TransformationType.TRANSLATION and not self.language_type:
            self.logger.error("Language type required for translation transformation")
            return False
        
        # Validate language complexity
        if self.language_complexity and not 1 <= self.language_complexity <= 10:
            self.logger.error(f"Language complexity must be 1-10, got {self.language_complexity}")
            return False
        
        return True