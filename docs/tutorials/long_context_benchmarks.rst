Long Context Benchmarks Tutorial
=================================

This comprehensive tutorial demonstrates how to use ScrambleBench's long context benchmarks to evaluate LLMs on document-based tasks in a contamination-resistant manner. Long context benchmarks transform documents and Q&A pairs while preserving semantic meaning and answer-document alignment.

.. contents:: Table of Contents
   :depth: 2
   :local:

Overview
--------

Long context benchmarks address contamination in document-based evaluation tasks by systematically transforming source documents while maintaining the logical relationships needed to answer questions. Unlike simple text substitution, these benchmarks require sophisticated understanding of document structure, semantic preservation, and Q&A alignment.

**Key Capabilities:**

* **Document Transformation**: Multiple strategies for modifying long documents
* **Q&A Alignment**: Intelligent question-answer pair transformation
* **Semantic Preservation**: Maintains meaning while changing surface form
* **Multi-Modal Support**: Handles extractive, abstractive, and multiple-choice questions
* **Scalable Processing**: Efficient handling of documents up to 100K+ tokens

Why Long Context Benchmarks Matter
----------------------------------

Traditional short-form benchmarks may not capture a model's ability to:

* Maintain coherence across long passages
* Track multiple entities and relationships
* Integrate information from different document sections
* Handle complex document structures (tables, lists, hierarchies)

Long context benchmarks reveal these capabilities through contamination-resistant evaluation.

**Common Long Context Tasks:**

* **Reading Comprehension**: Answer questions about long passages
* **Document Summarization**: Extract key information from documents
* **Information Extraction**: Find specific facts across long texts
* **Multi-Hop Reasoning**: Combine information from multiple document sections

Understanding Document Transformation
-------------------------------------

ScrambleBench uses several strategies to transform documents while preserving their question-answerable content:

Translation-Based Transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Transform documents using constructed languages while maintaining logical structure.

**How it works:**
- Applies language transformation to document text
- Transforms questions and answers consistently
- Preserves entity relationships and factual content
- Maintains document structure (paragraphs, headings, etc.)

**Example:**

.. code-block:: text

   Original Document:
   "Machine learning is a subset of artificial intelligence (AI) that enables 
   computers to learn and improve from experience without being explicitly 
   programmed. It focuses on developing algorithms that can access data and 
   use it to learn for themselves."

   Transformed Document (Phonetic):
   "Maghine learning ms a subset of artimicial mntelligenge (AM) zhat enables 
   gomputers to learn and mmbrove from experienke without being expliqitly 
   brogrammed. Mt foguses on developing algormthms zhat gan agcess data and 
   use mt to learn for zhemselves."

   Original Question: "What is machine learning?"
   Transformed Question: "Xhat ms maghine learning?"

**Best for:** Testing core language understanding while controlling for memorization.

Paraphrase-Based Transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rewrite document content while preserving meaning and factual accuracy.

**How it works:**
- Uses advanced language models to paraphrase content
- Maintains factual accuracy and logical flow
- Preserves technical terms and proper nouns
- Adjusts questions to match paraphrased content

**Example:**

.. code-block:: text

   Original:
   "The company was founded in 1998 by two Stanford PhD students who 
   developed a novel search algorithm called PageRank."

   Paraphrased:
   "Two doctoral candidates from Stanford University established the 
   organization in 1998 after creating an innovative search methodology 
   known as PageRank."

   Question Adjustment:
   Original: "When was the company founded?"
   Adjusted: "When was the organization established?"

**Best for:** Testing semantic understanding while changing surface form significantly.

Structural Reordering
~~~~~~~~~~~~~~~~~~~~

Reorganize document sections while maintaining logical coherence.

**How it works:**
- Identifies logical document sections
- Reorders paragraphs/sections systematically
- Updates references and cross-links appropriately
- Adjusts questions to reflect new structure

**Example:**

.. code-block:: text

   Original Structure:
   1. Introduction to AI
   2. Machine Learning Basics  
   3. Deep Learning Applications
   4. Future Prospects

   Reordered Structure:
   1. Future Prospects
   2. Introduction to AI
   3. Deep Learning Applications
   4. Machine Learning Basics

   Question Updates:
   "In the third section, what applications are discussed?"
   → "In the second section, what applications are discussed?"

**Best for:** Testing document navigation and structural understanding.

Hybrid Transformation
~~~~~~~~~~~~~~~~~~~

Combines multiple transformation strategies for maximum contamination resistance.

**How it works:**
- Applies translation to some sections
- Paraphrases other sections
- Reorders document structure
- Creates comprehensive transformation

**Example:**

.. code-block:: text

   Section 1: Translated (phonetic transformation)
   Section 2: Paraphrased (semantic preservation)
   Section 3: Reordered (moved from original Section 5)
   Section 4: Hybrid (paraphrased + translated)

**Best for:** Maximum robustness testing and comprehensive evaluation.

Setting Up Long Context Benchmarks
-----------------------------------

Basic Setup
~~~~~~~~~~~

Let's create a simple long context benchmark:

.. code-block:: python

   from scramblebench import LongContextBenchmark
   from scramblebench.llm import OpenRouterClient
   from scramblebench.longcontext.document_transformer import TransformationType
   import json

   # Create long context dataset
   long_context_data = [{
       "id": "doc1",
       "title": "Artificial Intelligence Overview",
       "document": """
       Artificial Intelligence (AI) represents one of the most significant 
       technological advances of the 21st century. It encompasses various 
       subfields including machine learning, natural language processing, 
       computer vision, and robotics.
       
       Machine learning, a core component of AI, enables systems to 
       automatically improve their performance through experience. This 
       field has revolutionized industries from healthcare to finance, 
       providing unprecedented capabilities for data analysis and prediction.
       
       Deep learning, a subset of machine learning, uses neural networks 
       with multiple layers to model and understand complex patterns in data. 
       This approach has achieved remarkable success in image recognition, 
       speech processing, and language understanding tasks.
       """,
       "questions": [
           "What does AI encompass?",
           "How does machine learning improve performance?",
           "What is deep learning?"
       ],
       "answers": [
           "AI encompasses machine learning, natural language processing, computer vision, and robotics",
           "Machine learning enables systems to automatically improve through experience",
           "Deep learning uses neural networks with multiple layers to understand complex patterns"
       ]
   }]

   # Save dataset
   with open("ai_overview.json", "w") as f:
       json.dump(long_context_data, f, indent=2)

   # Create long context benchmark
   benchmark = LongContextBenchmark(
       dataset_name="ai_overview.json",
       transformation_type=TransformationType.TRANSLATION,
       language_complexity=5,
       preserve_structure=True,
       seed=42
   )

   # Initialize model
   model = OpenRouterClient(
       model_name="openai/gpt-4",
       api_key="your-openrouter-key"
   )

   # Run evaluation
   results = benchmark.run(model, num_samples=1)
   print(f"Accuracy: {results.score:.2%}")

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

For sophisticated document transformation:

.. code-block:: python

   from scramblebench.longcontext.config import DocumentConfig, QAConfig

   # Configure document transformation
   doc_config = DocumentConfig(
       preserve_numbers=True,         # Keep numerical values
       preserve_entities=True,        # Maintain proper nouns
       preserve_structure=True,       # Keep document organization
       min_section_length=100,        # Minimum section size for reordering
       paraphrase_probability=0.7,    # 70% of sentences get paraphrased
       reorder_probability=0.3        # 30% chance of section reordering
   )

   # Configure Q&A transformation
   qa_config = QAConfig(
       question_transformation_rate=0.9,  # Transform 90% of questions
       answer_alignment_threshold=0.8,    # Require 80% semantic similarity
       preserve_answer_type=True,         # Maintain extractive/abstractive distinction
       max_answer_length_change=0.2       # Allow 20% length variation in answers
   )

   # Create advanced benchmark
   benchmark = LongContextBenchmark(
       dataset_name="complex_documents.json",
       transformation_type=TransformationType.HYBRID,
       document_config=doc_config,
       qa_config=qa_config,
       language_complexity=7,
       seed=42
   )

Working with Different Q&A Types
--------------------------------

Extractive Q&A
~~~~~~~~~~~~~~~

Questions where answers are exact spans from the document.

.. code-block:: python

   extractive_data = [{
       "id": "extract1",
       "document": "The Eiffel Tower was built in 1889 by Gustave Eiffel for the Paris Exposition.",
       "questions": ["When was the Eiffel Tower built?"],
       "answers": ["1889"],  # Exact span from document
       "answer_types": ["extractive"]
   }]

   # Extractive benchmarks preserve span alignment
   benchmark = LongContextBenchmark(
       dataset_name="extractive_qa.json",
       transformation_type=TransformationType.TRANSLATION,
       preserve_spans=True,  # Maintain answer-document alignment
       language_complexity=5
   )

Abstractive Q&A
~~~~~~~~~~~~~~~

Questions requiring synthesis or reformulation of document content.

.. code-block:: python

   abstractive_data = [{
       "id": "abstract1", 
       "document": """
       Climate change is caused by increased greenhouse gas emissions from human 
       activities. The primary gases include carbon dioxide from fossil fuel 
       combustion and methane from agriculture. These gases trap heat in the 
       atmosphere, leading to global temperature rise.
       """,
       "questions": ["What causes climate change?"],
       "answers": ["Human activities that increase greenhouse gas emissions, primarily CO2 and methane"],
       "answer_types": ["abstractive"]
   }]

   # Abstractive benchmarks focus on semantic preservation
   benchmark = LongContextBenchmark(
       dataset_name="abstractive_qa.json", 
       transformation_type=TransformationType.PARAPHRASE,
       preserve_semantics=True,
       paraphrase_strength=0.7
   )

Multiple Choice Q&A
~~~~~~~~~~~~~~~~~~~

Questions with predetermined answer options.

.. code-block:: python

   multiple_choice_data = [{
       "id": "mcq1",
       "document": "Photosynthesis converts carbon dioxide and water into glucose using sunlight energy.",
       "questions": ["What does photosynthesis produce?"],
       "choices": [
           ["oxygen", "glucose", "carbon dioxide", "water"]
       ],
       "answers": ["glucose"],
       "answer_types": ["multiple_choice"]
   }]

   # Multiple choice requires choice transformation
   benchmark = LongContextBenchmark(
       dataset_name="mcq_data.json",
       transformation_type=TransformationType.TRANSLATION,
       transform_choices=True,  # Transform answer choices too
       language_complexity=4
   )

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

Comprehensive CLI support for long context evaluation:

**Generate Transformed Documents:**

.. code-block:: bash

   # Transform a single document
   scramblebench longcontext transform-document \
     --input "original_doc.json" \
     --output "transformed_doc.json" \
     --transformation-type "paraphrase" \
     --preserve-structure \
     --preserve-entities

   # Batch transform multiple documents  
   scramblebench longcontext batch-transform \
     --input-dir "docs/" \
     --output-dir "transformed_docs/" \
     --transformation-type "hybrid" \
     --language-complexity 6

**Run Long Context Evaluation:**

.. code-block:: bash

   # Single document evaluation
   scramblebench longcontext evaluate \
     --documents "research_papers.json" \
     --models "openai/gpt-4" \
     --transformation-type "translation" \
     --language-complexity 5 \
     --experiment-name "research_comprehension"

   # Multi-model comparison
   scramblebench longcontext evaluate \
     --documents "legal_docs.json,medical_texts.json" \
     --models "openai/gpt-4,anthropic/claude-3-sonnet,meta-llama/llama-2-70b-chat" \
     --transformation-types "translation,paraphrase,hybrid" \
     --complexities "3,5,7" \
     --experiment-name "professional_document_understanding" \
     --max-samples 50

Working with Large Documents
----------------------------

Memory Management
~~~~~~~~~~~~~~~~~

For documents exceeding model context limits:

.. code-block:: python

   from scramblebench.longcontext.chunking import DocumentChunker, ChunkingStrategy

   # Configure document chunking
   chunker = DocumentChunker(
       max_chunk_size=8000,  # Maximum tokens per chunk
       overlap_size=200,     # Overlap between chunks
       chunking_strategy=ChunkingStrategy.SEMANTIC,  # Chunk by meaning
       preserve_paragraphs=True
   )

   # Process large document
   large_doc_benchmark = LongContextBenchmark(
       dataset_name="large_documents.json",
       transformation_type=TransformationType.HYBRID,
       document_chunker=chunker,
       merge_chunk_results=True  # Combine results across chunks
   )

Sliding Window Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate models on different document sections:

.. code-block:: python

   from scramblebench.longcontext.evaluation import SlidingWindowEvaluator

   # Configure sliding window
   evaluator = SlidingWindowEvaluator(
       window_size=4000,    # Tokens per window
       stride=2000,         # Overlap between windows
       aggregation_strategy="weighted_average"
   )

   # Evaluate across document sections
   results = evaluator.evaluate(
       benchmark=benchmark,
       model=model,
       num_samples=20
   )

   # Analyze positional effects
   position_analysis = evaluator.analyze_positional_effects(results)
   print(f"Early position accuracy: {position_analysis['early']:.2%}")
   print(f"Middle position accuracy: {position_analysis['middle']:.2%}")
   print(f"Late position accuracy: {position_analysis['late']:.2%}")

Practical Evaluation Workflows
-------------------------------

Research Paper Comprehension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: Evaluating models on academic paper understanding.

.. code-block:: python

   def research_paper_evaluation(paper_directory, models, output_dir):
       """Comprehensive research paper comprehension evaluation."""
       
       import os
       from pathlib import Path
       
       results = {}
       
       # Load research papers
       papers = []
       for paper_file in os.listdir(paper_directory):
           if paper_file.endswith('.json'):
               with open(os.path.join(paper_directory, paper_file)) as f:
                   paper_data = json.load(f)
                   papers.append(paper_data)
       
       # Test different transformation strategies
       transformations = [
           TransformationType.TRANSLATION,
           TransformationType.PARAPHRASE, 
           TransformationType.STRUCTURAL,
           TransformationType.HYBRID
       ]
       
       for transformation in transformations:
           transformation_results = {}
           
           # Create benchmark for this transformation
           benchmark = LongContextBenchmark(
               dataset_name=paper_directory,
               transformation_type=transformation,
               language_complexity=6,
               preserve_citations=True,  # Keep academic references
               preserve_equations=True,  # Maintain mathematical content
               seed=42
           )
           
           # Evaluate each model
           for model_name in models:
               model = OpenRouterClient(model_name)
               
               # Run evaluation
               result = benchmark.run(model, num_samples=len(papers))
               transformation_results[model_name] = result
               
               print(f"Completed: {transformation.name} with {model_name}")
           
           results[transformation.name] = transformation_results
       
       # Save comprehensive results
       output_path = Path(output_dir)
       output_path.mkdir(exist_ok=True)
       
       with open(output_path / "research_paper_results.json", "w") as f:
           json.dump(results, f, indent=2, default=str)
       
       return results

Legal Document Analysis
~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: Testing understanding of legal texts and contracts.

.. code-block:: python

   def legal_document_evaluation(legal_corpus, model, specializations):
       """Evaluate legal document comprehension across specializations."""
       
       results = {}
       
       for specialization in specializations:  # e.g., ['contracts', 'torts', 'constitutional']
           specialization_docs = [doc for doc in legal_corpus 
                                 if doc.get('specialization') == specialization]
           
           # Create specialized benchmark
           benchmark = LongContextBenchmark(
               dataset_name=specialization_docs,
               transformation_type=TransformationType.PARAPHRASE,
               preserve_legal_terms=True,     # Keep legal terminology
               preserve_citations=True,       # Maintain case references
               paraphrase_strength=0.6,       # Moderate paraphrasing
               language_complexity=5
           )
           
           # Run evaluation
           result = benchmark.run(model, num_samples=len(specialization_docs))
           
           # Analyze by question type
           question_type_analysis = {}
           for pred in result.predictions:
               q_type = pred.metadata.get('question_type', 'general')
               if q_type not in question_type_analysis:
                   question_type_analysis[q_type] = []
               question_type_analysis[q_type].append(pred.correct)
           
           # Calculate accuracy by question type
           for q_type, correctness in question_type_analysis.items():
               accuracy = sum(correctness) / len(correctness)
               question_type_analysis[q_type] = accuracy
           
           results[specialization] = {
               'overall_accuracy': result.score,
               'question_type_breakdown': question_type_analysis,
               'num_documents': len(specialization_docs)
           }
       
       return results

Technical Documentation Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: Evaluating models on technical manuals and documentation.

.. code-block:: python

   def technical_documentation_evaluation(tech_docs, models):
       """Test understanding of technical documentation."""
       
       # Configure for technical content
       doc_config = DocumentConfig(
           preserve_code_blocks=True,     # Keep code snippets intact
           preserve_technical_terms=True, # Maintain technical vocabulary
           preserve_diagrams=True,        # Keep ASCII diagrams/tables
           preserve_version_numbers=True  # Maintain version references
       )
       
       results = {}
       
       for model_name in models:
           model = OpenRouterClient(model_name)
           model_results = {}
           
           # Test different document types
           doc_types = ['API_reference', 'user_manual', 'troubleshooting', 'tutorial']
           
           for doc_type in doc_types:
               type_docs = [doc for doc in tech_docs if doc.get('type') == doc_type]
               
               if not type_docs:
                   continue
               
               benchmark = LongContextBenchmark(
                   dataset_name=type_docs,
                   transformation_type=TransformationType.HYBRID,
                   document_config=doc_config,
                   language_complexity=4,  # Lower complexity for technical content
                   seed=42
               )
               
               result = benchmark.run(model, num_samples=len(type_docs))
               
               # Analyze by difficulty
               difficulty_analysis = {}
               for pred in result.predictions:
                   difficulty = pred.metadata.get('difficulty', 'medium')
                   if difficulty not in difficulty_analysis:
                       difficulty_analysis[difficulty] = []
                   difficulty_analysis[difficulty].append(pred.correct)
               
               for difficulty, correctness in difficulty_analysis.items():
                   accuracy = sum(correctness) / len(correctness)
                   difficulty_analysis[difficulty] = accuracy
               
               model_results[doc_type] = {
                   'accuracy': result.score,
                   'difficulty_breakdown': difficulty_analysis
               }
           
           results[model_name] = model_results
       
       return results

Advanced Analysis Techniques
----------------------------

Document Understanding Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scramblebench.longcontext.metrics import (
       calculate_position_bias,
       measure_coherence_retention, 
       analyze_entity_tracking,
       compute_information_integration
   )

   def comprehensive_analysis(evaluation_results):
       """Perform comprehensive analysis of long context results."""
       
       analysis = {}
       
       # Position bias analysis
       position_bias = calculate_position_bias(evaluation_results)
       analysis['position_bias'] = {
           'early_advantage': position_bias['early'] > position_bias['late'],
           'bias_magnitude': abs(position_bias['early'] - position_bias['late']),
           'position_scores': position_bias
       }
       
       # Coherence analysis
       coherence = measure_coherence_retention(evaluation_results)
       analysis['coherence'] = {
           'overall_retention': coherence['retention_score'],
           'degradation_rate': coherence['degradation_per_1k_tokens'],
           'critical_length': coherence['critical_length']
       }
       
       # Entity tracking
       entity_tracking = analyze_entity_tracking(evaluation_results)
       analysis['entity_tracking'] = {
           'person_tracking_accuracy': entity_tracking['persons'],
           'organization_tracking_accuracy': entity_tracking['organizations'], 
           'location_tracking_accuracy': entity_tracking['locations'],
           'overall_entity_accuracy': entity_tracking['overall']
       }
       
       # Information integration
       integration = compute_information_integration(evaluation_results)
       analysis['information_integration'] = {
           'single_hop_accuracy': integration['single_hop'],
           'multi_hop_accuracy': integration['multi_hop'],
           'integration_penalty': integration['single_hop'] - integration['multi_hop']
       }
       
       return analysis

Error Pattern Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scramblebench.core.error_analysis import ErrorAnalyzer

   def analyze_long_context_errors(results):
       """Analyze common error patterns in long context evaluation."""
       
       analyzer = ErrorAnalyzer()
       
       # Categorize errors by type
       error_categories = {
           'factual_errors': [],
           'coherence_errors': [],
           'reference_errors': [],
           'comprehension_errors': []
       }
       
       for prediction in results.predictions:
           if not prediction.correct:
               error_type = analyzer.classify_error(
                   prediction.question,
                   prediction.expected_answer,
                   prediction.predicted_answer,
                   prediction.document_context
               )
               
               error_categories[error_type].append(prediction)
       
       # Analyze patterns within each category
       error_analysis = {}
       
       for category, errors in error_categories.items():
           if not errors:
               continue
           
           category_analysis = analyzer.analyze_category(errors)
           error_analysis[category] = {
               'frequency': len(errors) / len(results.predictions),
               'common_patterns': category_analysis['patterns'],
               'severity_distribution': category_analysis['severity'],
               'suggested_improvements': category_analysis['suggestions']
           }
       
       return error_analysis

Performance Optimization
------------------------

Efficient Document Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scramblebench.longcontext.optimization import (
       DocumentProcessor,
       OptimizationConfig
   )

   # Configure optimization
   opt_config = OptimizationConfig(
       batch_size=8,                    # Process 8 documents at once
       use_caching=True,               # Cache transformed documents
       parallel_processing=True,       # Use multiprocessing
       max_workers=4,                  # Limit concurrent workers
       memory_limit_gb=16             # Memory usage limit
   )

   # Create optimized processor
   processor = DocumentProcessor(
       transformation_type=TransformationType.HYBRID,
       optimization_config=opt_config
   )

   # Efficient batch processing
   def process_large_corpus(corpus_path, output_path):
       """Process large document corpus efficiently."""
       
       corpus = load_corpus(corpus_path)
       total_docs = len(corpus)
       
       processed_docs = []
       
       for batch_idx in range(0, total_docs, opt_config.batch_size):
           batch = corpus[batch_idx:batch_idx + opt_config.batch_size]
           
           # Process batch
           processed_batch = processor.process_batch(batch)
           processed_docs.extend(processed_batch)
           
           # Progress reporting
           completed = min(batch_idx + opt_config.batch_size, total_docs)
           print(f"Processed {completed}/{total_docs} documents")
           
           # Memory management
           if batch_idx % (opt_config.batch_size * 4) == 0:
               processor.clear_cache()
       
       # Save results
       save_corpus(processed_docs, output_path)
       return processed_docs

Streaming Evaluation
~~~~~~~~~~~~~~~~~~~

For very large document collections:

.. code-block:: python

   from scramblebench.longcontext.streaming import StreamingEvaluator

   def streaming_evaluation(document_stream, model, chunk_size=100):
       """Evaluate large document collections without loading everything into memory."""
       
       evaluator = StreamingEvaluator(
           model=model,
           chunk_size=chunk_size,
           save_intermediate_results=True
       )
       
       # Process documents in streaming fashion
       total_accuracy = 0
       total_processed = 0
       
       for document_chunk in document_stream:
           # Create benchmark for chunk
           benchmark = LongContextBenchmark(
               dataset_name=document_chunk,
               transformation_type=TransformationType.TRANSLATION,
               language_complexity=5
           )
           
           # Evaluate chunk
           chunk_results = benchmark.run(model, num_samples=len(document_chunk))
           
           # Update running statistics
           chunk_accuracy = chunk_results.score
           chunk_size = len(document_chunk)
           
           total_accuracy = ((total_accuracy * total_processed) + 
                           (chunk_accuracy * chunk_size)) / (total_processed + chunk_size)
           total_processed += chunk_size
           
           print(f"Processed {total_processed} documents. "
                 f"Running accuracy: {total_accuracy:.2%}")
       
       return total_accuracy, total_processed

Best Practices
--------------

Document Preparation
~~~~~~~~~~~~~~~~~~~

**1. Quality Assessment**

.. code-block:: python

   def assess_document_quality(documents):
       """Assess document quality for long context evaluation."""
       
       quality_metrics = {}
       
       for doc in documents:
           doc_id = doc['id']
           doc_text = doc['document']
           questions = doc['questions']
           answers = doc['answers']
           
           # Length analysis
           doc_length = len(doc_text.split())
           avg_question_length = sum(len(q.split()) for q in questions) / len(questions)
           avg_answer_length = sum(len(a.split()) for a in answers) / len(answers)
           
           # Complexity analysis
           sentences = doc_text.split('.')
           avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
           
           # Q&A coverage analysis
           answer_coverage = []
           for answer in answers:
               # Check if answer can be found in document
               coverage = calculate_answer_coverage(answer, doc_text)
               answer_coverage.append(coverage)
           
           quality_metrics[doc_id] = {
               'document_length': doc_length,
               'avg_sentence_length': avg_sentence_length,
               'avg_question_length': avg_question_length,
               'avg_answer_length': avg_answer_length,
               'answer_coverage': sum(answer_coverage) / len(answer_coverage),
               'qa_pairs': len(questions),
               'quality_score': calculate_quality_score(doc, questions, answers)
           }
       
       return quality_metrics

**2. Balanced Dataset Creation**

.. code-block:: python

   def create_balanced_long_context_dataset(documents, target_distribution):
       """Create balanced dataset across different document characteristics."""
       
       # Categorize documents
       categorized = {
           'short': [],    # < 1000 words
           'medium': [],   # 1000-5000 words  
           'long': [],     # 5000-15000 words
           'very_long': [] # > 15000 words
       }
       
       for doc in documents:
           word_count = len(doc['document'].split())
           
           if word_count < 1000:
               categorized['short'].append(doc)
           elif word_count < 5000:
               categorized['medium'].append(doc)
           elif word_count < 15000:
               categorized['long'].append(doc)
           else:
               categorized['very_long'].append(doc)
       
       # Sample according to target distribution
       balanced_dataset = []
       
       for category, target_count in target_distribution.items():
           available_docs = categorized[category]
           
           if len(available_docs) >= target_count:
               sampled = random.sample(available_docs, target_count)
           else:
               sampled = available_docs
               print(f"Warning: Only {len(available_docs)} documents in {category} category")
           
           balanced_dataset.extend(sampled)
       
       return balanced_dataset

Evaluation Design
~~~~~~~~~~~~~~~~~

**1. Systematic Testing**

.. code-block:: python

   def systematic_long_context_evaluation(
       documents, 
       models, 
       transformation_types, 
       complexity_levels,
       output_dir
   ):
       """Run systematic evaluation across all parameter combinations."""
       
       import itertools
       from pathlib import Path
       
       # Create output directory
       output_path = Path(output_dir)
       output_path.mkdir(exist_ok=True)
       
       # Generate all parameter combinations
       conditions = list(itertools.product(
           models, transformation_types, complexity_levels
       ))
       
       all_results = {}
       
       for model_name, transform_type, complexity in conditions:
           print(f"Testing: {model_name} with {transform_type.name} (complexity {complexity})")
           
           # Create benchmark
           benchmark = LongContextBenchmark(
               dataset_name=documents,
               transformation_type=transform_type,
               language_complexity=complexity,
               seed=42  # Consistent seed for reproducibility
           )
           
           # Initialize model
           model = OpenRouterClient(model_name)
           
           # Run evaluation
           try:
               result = benchmark.run(model, num_samples=len(documents))
               
               condition_key = f"{model_name}_{transform_type.name}_c{complexity}"
               all_results[condition_key] = {
                   'model': model_name,
                   'transformation': transform_type.name,
                   'complexity': complexity,
                   'accuracy': result.score,
                   'predictions': [
                       {
                           'question': p.question,
                           'expected': p.expected_answer,
                           'predicted': p.predicted_answer,
                           'correct': p.correct
                       } for p in result.predictions
                   ]
               }
               
               print(f"  Accuracy: {result.score:.2%}")
               
           except Exception as e:
               print(f"  Error: {e}")
               all_results[condition_key] = {'error': str(e)}
       
       # Save comprehensive results
       with open(output_path / "systematic_evaluation_results.json", "w") as f:
           json.dump(all_results, f, indent=2)
       
       return all_results

Troubleshooting
---------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Memory Errors with Large Documents**

.. code-block:: python

   # Solution: Use document chunking
   from scramblebench.longcontext.chunking import adaptive_chunking

   def handle_large_documents(documents, max_chunk_size=8000):
       """Handle documents that exceed model context limits."""
       
       processed_docs = []
       
       for doc in documents:
           doc_size = len(doc['document'].split())
           
           if doc_size > max_chunk_size:
               # Split document into manageable chunks
               chunks = adaptive_chunking(
                   document=doc['document'],
                   questions=doc['questions'],
                   answers=doc['answers'],
                   max_size=max_chunk_size,
                   overlap=200
               )
               
               processed_docs.extend(chunks)
           else:
               processed_docs.append(doc)
       
       return processed_docs

**2. Poor Transformation Quality**

.. code-block:: python

   # Solution: Validate transformations
   from scramblebench.longcontext.validation import TransformationValidator

   def validate_transformations(original_docs, transformed_docs):
       """Validate transformation quality."""
       
       validator = TransformationValidator()
       
       validation_results = []
       
       for orig, trans in zip(original_docs, transformed_docs):
           result = validator.validate(
               original_document=orig['document'],
               transformed_document=trans['document'],
               original_questions=orig['questions'],
               transformed_questions=trans['questions'],
               original_answers=orig['answers'],
               transformed_answers=trans['answers']
           )
           
           validation_results.append(result)
           
           # Report issues
           if result['semantic_similarity'] < 0.8:
               print(f"Warning: Low semantic similarity ({result['semantic_similarity']:.2f}) for doc {orig['id']}")
           
           if result['answer_alignment'] < 0.9:
               print(f"Warning: Poor answer alignment ({result['answer_alignment']:.2f}) for doc {orig['id']}")
       
       return validation_results

**3. Inconsistent Results Across Runs**

.. code-block:: python

   # Solution: Statistical validation
   def validate_result_consistency(benchmark, model, num_runs=5):
       """Validate consistency across multiple evaluation runs."""
       
       results = []
       
       for run in range(num_runs):
           # Use different seed for each run
           benchmark.seed = 42 + run
           result = benchmark.run(model, num_samples=50)
           results.append(result.score)
           print(f"Run {run + 1}: {result.score:.2%}")
       
       # Calculate statistics
       mean_accuracy = np.mean(results)
       std_accuracy = np.std(results)
       
       print(f"\nConsistency Analysis:")
       print(f"Mean Accuracy: {mean_accuracy:.2%}")
       print(f"Standard Deviation: {std_accuracy:.3f}")
       print(f"Coefficient of Variation: {std_accuracy/mean_accuracy:.3f}")
       
       # Flag high variance
       if std_accuracy > 0.05:  # More than 5% standard deviation
           print("WARNING: High variance detected. Consider:")
           print("- Increasing sample size")
           print("- Checking transformation determinism") 
           print("- Validating model temperature settings")
       
       return {
           'mean': mean_accuracy,
           'std': std_accuracy,
           'all_runs': results,
           'consistent': std_accuracy <= 0.05
       }

Integration with Research Workflows
----------------------------------

Academic Research Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def academic_research_pipeline(
       research_question,
       document_corpus, 
       models_to_evaluate,
       baseline_benchmarks,
       output_directory
   ):
       """Complete pipeline for academic research using long context benchmarks."""
       
       from datetime import datetime
       import matplotlib.pyplot as plt
       
       # Create timestamped output directory
       timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
       output_dir = Path(output_directory) / f"{research_question}_{timestamp}"
       output_dir.mkdir(parents=True, exist_ok=True)
       
       # Document research setup
       research_config = {
           'research_question': research_question,
           'document_corpus_size': len(document_corpus),
           'models_evaluated': models_to_evaluate,
           'evaluation_date': datetime.now().isoformat(),
           'baseline_benchmarks': baseline_benchmarks
       }
       
       with open(output_dir / "research_config.json", "w") as f:
           json.dump(research_config, f, indent=2)
       
       # Run comprehensive evaluation
       all_results = {}
       
       # Test different transformation strategies
       transformation_strategies = [
           TransformationType.TRANSLATION,
           TransformationType.PARAPHRASE,
           TransformationType.STRUCTURAL,
           TransformationType.HYBRID
       ]
       
       for transform_type in transformation_strategies:
           strategy_results = {}
           
           print(f"Evaluating transformation: {transform_type.name}")
           
           # Create benchmark
           benchmark = LongContextBenchmark(
               dataset_name=document_corpus,
               transformation_type=transform_type,
               language_complexity=6,
               seed=42
           )
           
           # Evaluate each model
           for model_name in models_to_evaluate:
               model = OpenRouterClient(model_name)
               
               # Run multiple evaluations for statistical significance
               model_results = []
               for run in range(3):  # 3 runs for significance
                   benchmark.seed = 42 + run
                   result = benchmark.run(model, num_samples=len(document_corpus))
                   model_results.append(result.score)
               
               # Calculate statistics
               mean_score = np.mean(model_results)
               std_score = np.std(model_results)
               
               strategy_results[model_name] = {
                   'mean_accuracy': mean_score,
                   'std_accuracy': std_score,
                   'individual_runs': model_results,
                   'confidence_interval': stats.t.interval(
                       0.95, len(model_results)-1,
                       loc=mean_score,
                       scale=stats.sem(model_results)
                   )
               }
               
               print(f"  {model_name}: {mean_score:.2%} (±{std_score:.3f})")
           
           all_results[transform_type.name] = strategy_results
       
       # Generate research report
       generate_academic_report(all_results, research_config, output_dir)
       
       # Create publication-ready visualizations
       create_research_visualizations(all_results, output_dir)
       
       print(f"Research pipeline completed. Results saved to: {output_dir}")
       
       return all_results, output_dir

Next Steps
----------

Now that you understand long context benchmarks:

1. **Experiment with Document Types**: Test different document structures and content types
2. **Advanced Transformations**: Explore hybrid transformation strategies
3. **Scale Up**: Use streaming evaluation for large document collections
4. **Custom Metrics**: Develop domain-specific evaluation metrics
5. **Integration**: Incorporate into your research or evaluation workflows

**Related Documentation:**

* :doc:`translation_benchmarks` - Constructed language techniques
* :doc:`custom_models` - Integrating specialized models
* :doc:`../user_guide/evaluation_pipeline` - Comprehensive evaluation workflows
* :doc:`../examples/advanced_usage` - Complex evaluation scenarios

**Community Resources:**

* GitHub Issues: Report bugs specific to long context evaluation
* GitHub Discussions: Share your document transformation strategies
* Research Papers: See academic applications in document understanding