System Architecture & Design
============================

ScrambleBench Architecture Overview
-----------------------------------

ScrambleBench is designed as a modular, extensible toolkit for contamination-resistant LLM evaluation. The architecture follows a layered approach that separates concerns while enabling flexible composition of evaluation pipelines.

**Core Architectural Principles:**

* **Separation of Concerns**: Clear boundaries between data processing, language generation, model interaction, and evaluation
* **Extensibility**: Plugin architecture for new benchmark types, language transformations, and model adapters
* **Reliability**: Robust error handling, retry mechanisms, and comprehensive logging
* **Performance**: Efficient data processing with support for batch operations and parallel execution
* **Reproducibility**: Deterministic operations with proper seed management and configuration tracking

**High-Level System Diagram:**

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────┐
    │                        ScrambleBench                            │
    │                                                                 │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
    │  │     CLI      │  │   Web API    │  │   Jupyter    │         │
    │  │   Interface  │  │  Interface   │  │  Interface   │         │
    │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
    │         │                 │                 │                 │
    │  ┌──────▼─────────────────▼─────────────────▼───────┐         │
    │  │              Evaluation Engine                    │         │
    │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │         │
    │  │  │Translation  │ │Long Context │ │  Custom     │ │         │
    │  │  │Benchmarks   │ │Benchmarks   │ │ Benchmarks  │ │         │
    │  │  └─────────────┘ └─────────────┘ └─────────────┘ │         │
    │  └──────────────────────┬────────────────────────────┘         │
    │                         │                                      │
    │  ┌──────────────────────▼────────────────────────────┐         │
    │  │                 Core Framework                     │         │
    │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │         │
    │  │  │  Benchmark  │ │  Evaluator  │ │  Reporter   │ │         │
    │  │  │    Base     │ │             │ │             │ │         │
    │  │  └─────────────┘ └─────────────┘ └─────────────┘ │         │
    │  └──────────────────────┬────────────────────────────┘         │
    │                         │                                      │
    │  ┌──────────────────────▼────────────────────────────┐         │
    │  │              Processing Layer                      │         │
    │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │         │
    │  │  │ Language    │ │  Document   │ │   Model     │ │         │
    │  │  │Generation   │ │Transformation│ │  Adapters   │ │         │
    │  │  └─────────────┘ └─────────────┘ └─────────────┘ │         │
    │  └──────────────────────┬────────────────────────────┘         │
    │                         │                                      │
    │  ┌──────────────────────▼────────────────────────────┐         │
    │  │                Utilities Layer                     │         │
    │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │         │
    │  │  │    Data     │ │    Config   │ │   Logging   │ │         │
    │  │  │   Loading   │ │Management   │ │ & Metrics   │ │         │
    │  │  └─────────────┘ └─────────────┘ └─────────────┘ │         │
    │  └─────────────────────────────────────────────────────┘         │
    └─────────────────────────────────────────────────────────────────┘

Core Framework Components
-------------------------

**BaseBenchmark Abstract Class**

The ``BaseBenchmark`` class serves as the foundation for all evaluation implementations, providing a standardized lifecycle and interface:

.. code-block:: python

    # Core benchmark lifecycle
    class BaseBenchmark(ABC):
        """
        Abstract base class defining the benchmark evaluation contract.
        
        Lifecycle:
        1. __init__() - Configuration and component initialization
        2. prepare_data() - Data loading and preprocessing
        3. run() - Orchestrated evaluation execution
            - get_evaluation_data() - Retrieve processed data
            - run_single_evaluation() - Per-item evaluation
            - compute_metrics() - Aggregate results
        4. save_result() - Persistence and reporting
        """
        
        def run(self, model, num_samples=None, save_results=True):
            """Orchestrate complete benchmark execution."""
            start_time = time.time()
            
            # Phase 1: Data preparation
            self.prepare_data()
            
            # Phase 2: Evaluation execution
            eval_data = self.get_evaluation_data(num_samples)
            individual_results = []
            
            for data_item in eval_data:
                result = self.run_single_evaluation(model, data_item)
                individual_results.append(result)
            
            # Phase 3: Metrics computation
            metrics = self.compute_metrics(individual_results)
            
            # Phase 4: Result packaging
            benchmark_result = BenchmarkResult(
                benchmark_name=self.name,
                model_name=getattr(model, 'name', str(model)),
                score=metrics.get('score', 0.0),
                metrics=metrics,
                metadata={'num_samples': len(eval_data)},
                duration=time.time() - start_time,
                timestamp=start_time
            )
            
            # Phase 5: Persistence
            if save_results:
                self.save_result(benchmark_result)
            
            return benchmark_result

**Key Design Patterns:**

1. **Template Method Pattern**: ``BaseBenchmark.run()`` defines the algorithm structure while subclasses implement specific steps
2. **Strategy Pattern**: Different evaluation strategies (translation, long context) inherit from the same base
3. **Observer Pattern**: Logging and metrics collection observe evaluation progress
4. **Factory Pattern**: Model adapters and language generators use factory methods for instantiation

Translation Architecture
------------------------

**Constructed Language Generation System**

The translation system implements a sophisticated language generation pipeline that creates artificial languages while preserving logical structure:

.. code-block:: text

    Language Generation Pipeline:
    
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │  Source Text    │───▶│  Tokenization   │───▶│  Linguistic     │
    │  Analysis       │    │  & Parsing      │    │  Structure      │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
               │                       │                       │
               ▼                       ▼                       ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │  Vocabulary     │    │  Grammar Rule   │    │  Semantic       │
    │  Extraction     │    │  Identification │    │  Relationship   │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
               │                       │                       │
               ▼                       ▼                       ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │  Constructed    │    │  Translation    │    │  Validation &   │
    │  Language       │───▶│  Mapping        │───▶│  Verification   │
    │  Generation     │    │  Creation       │    │                 │
    └─────────────────┘    └─────────────────┘    └─────────────────┘

**Language Type Implementations:**

.. code-block:: python

    class LanguageType(Enum):
        """Enumeration of supported constructed language types."""
        
        SUBSTITUTION = "substitution"    # Direct word substitution
        PHONETIC = "phonetic"           # Phonetic transformations  
        SCRAMBLED = "scrambled"         # Character/syllable scrambling
        AGGLUTINATIVE = "agglutinative" # Morphological agglutination
        SYNTHETIC = "synthetic"         # Fully synthetic grammar
        HYBRID = "hybrid"               # Multiple transformation types

    class LanguageGenerator:
        """Core language generation engine."""
        
        def generate_language(
            self, 
            name: str, 
            language_type: LanguageType, 
            complexity: int,
            vocab_size: int = 1000,
            seed: Optional[int] = None
        ) -> ConstructedLanguage:
            """
            Generate a constructed language with specified characteristics.
            
            Architecture:
            1. Vocabulary Analysis - Extract common words and patterns
            2. Phonetic Mapping - Create sound-preserving transformations
            3. Grammar Rules - Define syntactic transformation rules
            4. Consistency Validation - Ensure bijective mappings
            5. Complexity Scaling - Adjust difficulty based on parameters
            """
            
            # Initialize deterministic generation
            if seed:
                random.seed(seed)
                np.random.seed(seed)
            
            # Core generation pipeline
            base_vocabulary = self._extract_base_vocabulary(vocab_size)
            phonetic_patterns = self._analyze_phonetic_patterns(base_vocabulary)
            transformation_rules = self._create_transformation_rules(
                language_type, complexity
            )
            
            # Generate language mappings
            language = ConstructedLanguage(
                name=name,
                language_type=language_type,
                complexity=complexity
            )
            
            # Apply transformations
            for word in base_vocabulary:
                transformed = self._apply_transformations(
                    word, transformation_rules, phonetic_patterns
                )
                language.add_mapping(word, transformed)
            
            # Validate consistency
            self._validate_language_consistency(language)
            
            return language

**Translation Preservation Properties:**

The translation system maintains several critical invariants:

1. **Bijective Mapping**: Every original text has exactly one translation and vice versa
2. **Structural Preservation**: Mathematical operators, logical connectors, and syntactic structure remain intact
3. **Deterministic Generation**: Same seed produces identical language mappings
4. **Compositionality**: Complex expressions translate compositionally from components

.. code-block:: python

    class TranslationInvariants:
        """Validation of translation preservation properties."""
        
        @staticmethod
        def validate_bijective_mapping(language: ConstructedLanguage):
            """Ensure one-to-one correspondence between original and translated terms."""
            forward_mappings = set(language.mappings.values())
            reverse_mappings = set(language.reverse_mappings.keys())
            
            assert forward_mappings == reverse_mappings
            assert len(language.mappings) == len(language.reverse_mappings)
        
        @staticmethod
        def validate_structural_preservation(original: str, translated: str):
            """Verify that logical and mathematical structure is preserved."""
            # Mathematical operators must be preserved
            math_ops = ['+', '-', '*', '/', '=', '<', '>', '≤', '≥']
            for op in math_ops:
                assert original.count(op) == translated.count(op)
            
            # Logical connectors must be preserved
            logical_ops = ['and', 'or', 'not', 'if', 'then', 'iff']
            for op in logical_ops:
                # Check for preserved logical structure (may be translated)
                original_logical_count = len(re.findall(r'\b' + op + r'\b', original))
                # Implementation would check for translated equivalents
        
        @staticmethod
        def validate_round_trip_consistency(language: ConstructedLanguage, text: str):
            """Ensure perfect round-trip translation."""
            translated = language.translate(text)
            reconstructed = language.reverse_translate(translated)
            assert text == reconstructed

Long Context Architecture
-------------------------

**Document Transformation Pipeline**

Long context evaluation requires sophisticated document transformation that preserves semantic content while changing surface form:

.. code-block:: text

    Document Transformation Pipeline:
    
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   Source        │───▶│   Document      │───▶│   Semantic      │
    │   Document      │    │   Parsing       │    │   Analysis      │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
              │                       │                       │
              ▼                       ▼                       ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   Question/     │    │   Entity        │    │   Relationship  │
    │   Answer        │    │   Extraction    │    │   Mapping       │
    │   Extraction    │    │                 │    │                 │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
              │                       │                       │
              ▼                       ▼                       ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   Transform     │    │   Answer        │    │   Validation    │
    │   Document      │───▶│   Alignment     │───▶│   & Quality     │
    │   Content       │    │   Tracking      │    │   Assurance     │
    └─────────────────┘    └─────────────────┘    └─────────────────┘

**Document Transformer Implementation:**

.. code-block:: python

    class DocumentTransformer:
        """Transforms long context documents while preserving answer extractability."""
        
        def __init__(self, transformation_type: str = "semantic_preserving"):
            self.transformation_type = transformation_type
            self.entity_tracker = EntityTracker()
            self.answer_aligner = AnswerAligner()
            self.consistency_validator = ConsistencyValidator()
        
        def transform_document(
            self, 
            document: str, 
            questions: List[Dict],
            preserve_entities: bool = True
        ) -> TransformedDocument:
            """
            Transform document while maintaining answer extractability.
            
            Architecture:
            1. Parse document structure (paragraphs, sentences, entities)
            2. Extract and map critical entities and relationships
            3. Apply transformations while preserving semantic content
            4. Align answers with transformed content
            5. Validate answer extractability
            """
            
            # Phase 1: Document analysis
            doc_structure = self._parse_document_structure(document)
            entities = self.entity_tracker.extract_entities(document)
            answer_spans = self._extract_answer_spans(document, questions)
            
            # Phase 2: Transformation planning
            transformation_plan = self._create_transformation_plan(
                doc_structure, entities, answer_spans
            )
            
            # Phase 3: Content transformation
            transformed_content = self._apply_transformations(
                document, transformation_plan
            )
            
            # Phase 4: Answer alignment
            aligned_answers = self.answer_aligner.align_answers(
                original_answers=answer_spans,
                transformed_document=transformed_content,
                transformation_mappings=transformation_plan.mappings
            )
            
            # Phase 5: Validation
            self.consistency_validator.validate_transformation(
                original=document,
                transformed=transformed_content,
                answer_alignment=aligned_answers
            )
            
            return TransformedDocument(
                original_document=document,
                transformed_document=transformed_content,
                transformation_mappings=transformation_plan.mappings,
                answer_alignments=aligned_answers,
                metadata={
                    'transformation_type': self.transformation_type,
                    'entity_count': len(entities),
                    'preservation_score': self._compute_preservation_score(
                        document, transformed_content
                    )
                }
            )

**Answer Alignment System:**

Critical for long context evaluation is maintaining the extractability of correct answers from transformed documents:

.. code-block:: python

    class AnswerAligner:
        """Ensures answers remain extractable after document transformation."""
        
        def align_answers(
            self,
            original_answers: List[AnswerSpan],
            transformed_document: str,
            transformation_mappings: Dict[str, str]
        ) -> List[AlignedAnswer]:
            """
            Map original answer spans to transformed document locations.
            
            Approach:
            1. Semantic Embedding - Use embeddings to find semantically similar spans
            2. Entity Tracking - Follow entity transformations through mappings
            3. Context Matching - Match surrounding context patterns
            4. Validation - Verify answer correctness is preserved
            """
            
            aligned_answers = []
            
            for answer_span in original_answers:
                # Method 1: Direct mapping lookup
                if answer_span.text in transformation_mappings:
                    transformed_text = transformation_mappings[answer_span.text]
                    new_span = self._find_span_in_document(
                        transformed_document, transformed_text
                    )
                    
                # Method 2: Semantic similarity search
                else:
                    new_span = self._find_semantically_similar_span(
                        answer_span, transformed_document
                    )
                
                # Method 3: Context-based alignment
                if new_span is None:
                    new_span = self._align_by_context(
                        answer_span, transformed_document, transformation_mappings
                    )
                
                # Validate alignment quality
                alignment_confidence = self._compute_alignment_confidence(
                    answer_span, new_span, transformed_document
                )
                
                aligned_answers.append(AlignedAnswer(
                    original_span=answer_span,
                    transformed_span=new_span,
                    confidence=alignment_confidence,
                    alignment_method=self._get_alignment_method(new_span)
                ))
            
            return aligned_answers

Model Adapter Architecture
--------------------------

**Unified Model Interface**

ScrambleBench supports diverse LLM providers through a unified adapter interface:

.. code-block:: python

    class ModelInterface(ABC):
        """Abstract interface for all LLM integrations."""
        
        @abstractmethod
        async def generate(
            self, 
            prompt: str, 
            max_tokens: Optional[int] = None,
            temperature: float = 0.0,
            **kwargs
        ) -> str:
            """Generate text completion for given prompt."""
            pass
        
        @abstractmethod
        def get_model_info(self) -> ModelInfo:
            """Return model capabilities and configuration."""
            pass
        
        @property
        @abstractmethod
        def max_context_length(self) -> int:
            """Maximum context length supported by model."""
            pass

**OpenRouter Integration:**

.. code-block:: python

    class OpenRouterClient(ModelInterface):
        """OpenRouter API client with rate limiting and error handling."""
        
        def __init__(
            self,
            model_name: str,
            api_key: str,
            rate_limit: float = 1.0,
            max_retries: int = 3,
            timeout: float = 30.0
        ):
            self.model_name = model_name
            self.api_key = api_key
            self.rate_limiter = RateLimiter(rate_limit)
            self.retry_handler = RetryHandler(max_retries)
            self.timeout = timeout
            
            # Initialize HTTP session with connection pooling
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout),
                connector=aiohttp.TCPConnector(limit=10)
            )
        
        async def generate(self, prompt: str, **kwargs) -> str:
            """Generate completion with comprehensive error handling."""
            
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            # Prepare request
            request_data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.0)
            }
            
            # Execute with retry logic
            response = await self.retry_handler.execute(
                self._make_request, request_data
            )
            
            return self._extract_response_text(response)
        
        async def _make_request(self, request_data: Dict) -> Dict:
            """Execute HTTP request with error handling."""
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with self.session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=request_data,
                headers=headers
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"OpenRouter API error: {error_text}"
                    )
                
                return await response.json()

**Model Adapter Factory:**

.. code-block:: python

    class ModelAdapterFactory:
        """Factory for creating appropriate model adapters."""
        
        _adapters = {
            'openrouter': OpenRouterClient,
            'huggingface': HuggingFaceAdapter,
            'openai': OpenAIAdapter,
            'anthropic': AnthropicAdapter
        }
        
        @classmethod
        def create_adapter(
            cls,
            provider: str,
            model_name: str,
            **kwargs
        ) -> ModelInterface:
            """Create model adapter for specified provider."""
            
            if provider not in cls._adapters:
                raise ValueError(f"Unsupported provider: {provider}")
            
            adapter_class = cls._adapters[provider]
            return adapter_class(model_name=model_name, **kwargs)
        
        @classmethod
        def register_adapter(cls, name: str, adapter_class: Type[ModelInterface]):
            """Register custom model adapter."""
            cls._adapters[name] = adapter_class

Data Processing Architecture
----------------------------

**Data Loading Pipeline**

ScrambleBench supports multiple data formats and sources through a unified loading interface:

.. code-block:: python

    class DataLoader:
        """Unified data loading interface supporting multiple formats."""
        
        def __init__(self, config: Config, logger: logging.Logger):
            self.config = config
            self.logger = logger
            self.format_handlers = {
                '.json': self._load_json,
                '.jsonl': self._load_jsonl,
                '.csv': self._load_csv,
                '.tsv': self._load_tsv,
                '.parquet': self._load_parquet
            }
        
        def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
            """Load dataset from file or identifier."""
            
            # Handle built-in datasets
            if dataset_path in BUILTIN_DATASETS:
                return self._load_builtin_dataset(dataset_path)
            
            # Handle file paths
            path = Path(dataset_path)
            if not path.exists():
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            
            # Route to appropriate handler
            suffix = path.suffix.lower()
            if suffix not in self.format_handlers:
                raise ValueError(f"Unsupported format: {suffix}")
            
            handler = self.format_handlers[suffix]
            data = handler(path)
            
            # Validate data structure
            self._validate_dataset(data)
            
            return data
        
        def _validate_dataset(self, data: List[Dict]) -> None:
            """Validate dataset structure and required fields."""
            if not data:
                raise ValueError("Dataset is empty")
            
            required_fields = self.config.get('required_fields', ['question', 'answer'])
            
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    raise ValueError(f"Item {i} is not a dictionary")
                
                for field in required_fields:
                    if field not in item:
                        raise ValueError(f"Item {i} missing required field: {field}")

**Metrics Computation System**

.. code-block:: python

    class MetricsComputer(ABC):
        """Abstract base for metrics computation."""
        
        @abstractmethod
        def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Compute aggregate metrics from evaluation results."""
            pass

    class TranslationMetricsComputer(MetricsComputer):
        """Metrics computation for translation benchmarks."""
        
        def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Compute comprehensive translation benchmark metrics."""
            
            if not results:
                return self._empty_metrics()
            
            # Basic accuracy metrics
            correct_count = sum(1 for r in results if r.get('correct', False))
            total_count = len(results)
            accuracy = correct_count / total_count
            
            # Response quality metrics
            avg_response_length = np.mean([
                len(r.get('response', '')) for r in results
            ])
            
            response_times = [r.get('response_time', 0) for r in results]
            avg_response_time = np.mean(response_times)
            
            # Translation-specific metrics
            translation_quality = self._compute_translation_quality(results)
            consistency_score = self._compute_consistency_score(results)
            
            # Statistical confidence
            confidence_interval = self._compute_confidence_interval(
                accuracy, total_count
            )
            
            return {
                'score': accuracy,  # Primary metric
                'accuracy': accuracy,
                'correct_count': correct_count,
                'total_count': total_count,
                'avg_response_length': avg_response_length,
                'avg_response_time': avg_response_time,
                'translation_quality': translation_quality,
                'consistency_score': consistency_score,
                'confidence_interval': confidence_interval,
                'statistical_significance': self._compute_significance(results)
            }

Configuration & Extension Architecture
-------------------------------------

**Configuration System**

ScrambleBench uses a hierarchical configuration system supporting YAML files, environment variables, and programmatic configuration:

.. code-block:: python

    class Config:
        """Hierarchical configuration management."""
        
        def __init__(self, config_data: Optional[Dict] = None):
            self._data = {}
            self._defaults = self._load_defaults()
            
            # Load configuration hierarchy
            self._load_from_defaults()
            self._load_from_environment()
            self._load_from_files()
            
            if config_data:
                self._data.update(config_data)
        
        def get(self, key: str, default: Any = None) -> Any:
            """Get configuration value with dot notation support."""
            keys = key.split('.')
            value = self._data
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
        
        def _load_from_environment(self):
            """Load configuration from environment variables."""
            env_prefix = "SCRAMBLEBENCH_"
            
            for key, value in os.environ.items():
                if key.startswith(env_prefix):
                    config_key = key[len(env_prefix):].lower().replace('_', '.')
                    self._set_nested_key(config_key, self._parse_env_value(value))

**Plugin Architecture**

ScrambleBench supports extensions through a plugin system:

.. code-block:: python

    class PluginManager:
        """Manage plugins and extensions."""
        
        def __init__(self):
            self._plugins = {}
            self._hooks = defaultdict(list)
        
        def register_plugin(self, name: str, plugin: Plugin):
            """Register a plugin."""
            self._plugins[name] = plugin
            
            # Register plugin hooks
            for hook_name in plugin.get_hooks():
                self._hooks[hook_name].append(plugin)
        
        def get_plugin(self, name: str) -> Optional[Plugin]:
            """Get registered plugin by name."""
            return self._plugins.get(name)
        
        def execute_hook(self, hook_name: str, *args, **kwargs):
            """Execute all plugins registered for a hook."""
            results = []
            
            for plugin in self._hooks[hook_name]:
                try:
                    result = plugin.execute_hook(hook_name, *args, **kwargs)
                    results.append(result)
                except Exception as e:
                    logging.error(f"Plugin {plugin.name} hook {hook_name} failed: {e}")
            
            return results

    class Plugin(ABC):
        """Abstract base class for plugins."""
        
        @property
        @abstractmethod
        def name(self) -> str:
            """Plugin name."""
            pass
        
        @abstractmethod
        def get_hooks(self) -> List[str]:
            """Return list of hook names this plugin handles."""
            pass
        
        @abstractmethod
        def execute_hook(self, hook_name: str, *args, **kwargs) -> Any:
            """Execute plugin hook."""
            pass

Performance & Scalability Design
--------------------------------

**Async Processing Architecture**

ScrambleBench leverages asynchronous processing for improved performance:

.. code-block:: python

    class AsyncEvaluationEngine:
        """Asynchronous evaluation execution engine."""
        
        def __init__(self, max_concurrent: int = 10):
            self.max_concurrent = max_concurrent
            self.semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_batch_evaluation(
            self,
            benchmark: BaseBenchmark,
            model: ModelInterface,
            eval_data: List[Any]
        ) -> List[Dict[str, Any]]:
            """Execute evaluation batch with controlled concurrency."""
            
            # Create evaluation tasks
            tasks = [
                self._evaluate_single_item(benchmark, model, item)
                for item in eval_data
            ]
            
            # Execute with controlled concurrency
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions and filter results
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.error(f"Evaluation {i} failed: {result}")
                else:
                    valid_results.append(result)
            
            return valid_results
        
        async def _evaluate_single_item(
            self,
            benchmark: BaseBenchmark,
            model: ModelInterface,
            data_item: Any
        ) -> Dict[str, Any]:
            """Evaluate single item with semaphore control."""
            async with self.semaphore:
                return await asyncio.get_event_loop().run_in_executor(
                    None, benchmark.run_single_evaluation, model, data_item
                )

**Caching & Optimization**

.. code-block:: python

    class CacheManager:
        """Intelligent caching for expensive operations."""
        
        def __init__(self, cache_dir: Path, max_size_gb: float = 1.0):
            self.cache_dir = cache_dir
            self.max_size_bytes = int(max_size_gb * 1024**3)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        def get_cached_result(self, cache_key: str) -> Optional[Any]:
            """Retrieve cached result if available."""
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    logging.warning(f"Cache read failed for {cache_key}: {e}")
                    cache_file.unlink()  # Remove corrupted cache
            
            return None
        
        def cache_result(self, cache_key: str, result: Any) -> None:
            """Cache result with size management."""
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                
                # Manage cache size
                self._cleanup_cache_if_needed()
                
            except Exception as e:
                logging.error(f"Cache write failed for {cache_key}: {e}")
        
        def _cleanup_cache_if_needed(self):
            """Remove old cache files if size limit exceeded."""
            total_size = sum(f.stat().st_size for f in self.cache_dir.iterdir())
            
            if total_size > self.max_size_bytes:
                # Remove oldest files first
                files = sorted(
                    self.cache_dir.iterdir(),
                    key=lambda f: f.stat().st_mtime
                )
                
                for file in files:
                    file.unlink()
                    total_size -= file.stat().st_size
                    
                    if total_size <= self.max_size_bytes * 0.8:  # 20% buffer
                        break

Error Handling & Resilience
---------------------------

**Comprehensive Error Handling Strategy**

.. code-block:: python

    class ScrambleBenchError(Exception):
        """Base exception for ScrambleBench errors."""
        pass

    class ModelError(ScrambleBenchError):
        """Model-related errors."""
        pass

    class DataError(ScrambleBenchError):
        """Data processing errors."""
        pass

    class ConfigurationError(ScrambleBenchError):
        """Configuration-related errors."""
        pass

    class EvaluationError(ScrambleBenchError):
        """Evaluation execution errors."""
        pass

    class ErrorHandler:
        """Centralized error handling and recovery."""
        
        def __init__(self, logger: logging.Logger):
            self.logger = logger
            self.error_counts = defaultdict(int)
        
        def handle_error(
            self,
            error: Exception,
            context: str,
            recovery_action: Optional[Callable] = None
        ) -> bool:
            """Handle error with optional recovery."""
            
            error_type = type(error).__name__
            self.error_counts[error_type] += 1
            
            # Log error with context
            self.logger.error(
                f"Error in {context}: {error_type}: {str(error)}",
                exc_info=True
            )
            
            # Attempt recovery if provided
            if recovery_action:
                try:
                    recovery_action()
                    self.logger.info(f"Recovery successful for {context}")
                    return True
                except Exception as recovery_error:
                    self.logger.error(
                        f"Recovery failed for {context}: {recovery_error}"
                    )
            
            return False

**Retry Mechanism**

.. code-block:: python

    class RetryHandler:
        """Intelligent retry mechanism with exponential backoff."""
        
        def __init__(
            self,
            max_retries: int = 3,
            base_delay: float = 1.0,
            max_delay: float = 60.0,
            backoff_factor: float = 2.0
        ):
            self.max_retries = max_retries
            self.base_delay = base_delay
            self.max_delay = max_delay
            self.backoff_factor = backoff_factor
        
        async def execute(
            self,
            func: Callable,
            *args,
            retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
            **kwargs
        ) -> Any:
            """Execute function with retry logic."""
            
            last_exception = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                        
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == self.max_retries:
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.base_delay * (self.backoff_factor ** attempt),
                        self.max_delay
                    )
                    
                    logging.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    
                    await asyncio.sleep(delay)
            
            # Should never reach here, but for safety
            raise last_exception

Security & Privacy Architecture
------------------------------

**API Key Management**

.. code-block:: python

    class SecureCredentialManager:
        """Secure management of API keys and credentials."""
        
        def __init__(self):
            self.keyring_available = self._check_keyring_availability()
        
        def store_credential(self, service: str, key: str, value: str) -> None:
            """Store credential securely."""
            if self.keyring_available:
                import keyring
                keyring.set_password(f"scramblebench_{service}", key, value)
            else:
                # Fallback to environment variables with warning
                logging.warning(
                    "Keyring not available. Consider using environment variables "
                    "or system keyring for secure credential storage."
                )
        
        def get_credential(self, service: str, key: str) -> Optional[str]:
            """Retrieve credential securely."""
            # Try keyring first
            if self.keyring_available:
                import keyring
                credential = keyring.get_password(f"scramblebench_{service}", key)
                if credential:
                    return credential
            
            # Fallback to environment variables
            env_key = f"SCRAMBLEBENCH_{service.upper()}_{key.upper()}"
            return os.environ.get(env_key)

**Data Privacy Protection**

.. code-block:: python

    class PrivacyManager:
        """Manage data privacy and anonymization."""
        
        def __init__(self):
            self.anonymization_strategies = {
                'names': self._anonymize_names,
                'locations': self._anonymize_locations,
                'dates': self._anonymize_dates,
                'numbers': self._anonymize_numbers
            }
        
        def anonymize_data(
            self,
            data: List[Dict],
            strategies: List[str] = None
        ) -> List[Dict]:
            """Anonymize sensitive data in evaluation datasets."""
            
            if strategies is None:
                strategies = list(self.anonymization_strategies.keys())
            
            anonymized_data = []
            
            for item in data:
                anonymized_item = item.copy()
                
                for strategy in strategies:
                    if strategy in self.anonymization_strategies:
                        anonymized_item = self.anonymization_strategies[strategy](
                            anonymized_item
                        )
                
                anonymized_data.append(anonymized_item)
            
            return anonymized_data

Future Architecture Roadmap
---------------------------

**Planned Architectural Enhancements:**

1. **Distributed Evaluation**
   - Multi-node evaluation clusters
   - Kubernetes deployment support
   - Load balancing and auto-scaling

2. **Real-time Streaming**
   - WebSocket-based evaluation streaming
   - Live progress monitoring
   - Interactive evaluation dashboards

3. **Advanced Model Support**
   - Multi-modal model evaluation
   - Tool-using agent evaluation
   - Federated learning model support

4. **Enhanced Language Generation**
   - Neural language generation
   - Semantic equivalence validation
   - Cross-lingual evaluation support

5. **Intelligent Optimization**
   - Adaptive evaluation strategies
   - Automatic hyperparameter tuning
   - Predictive resource allocation

**Extension Points:**

The architecture provides several extension points for customization:

- **Custom Benchmarks**: Inherit from ``BaseBenchmark``
- **Custom Language Types**: Extend ``LanguageGenerator``
- **Custom Model Adapters**: Implement ``ModelInterface``
- **Custom Metrics**: Extend ``MetricsComputer``
- **Custom Transformations**: Implement transformation plugins
- **Custom Data Sources**: Extend ``DataLoader``

This modular architecture ensures ScrambleBench can evolve with the rapidly changing landscape of LLM evaluation while maintaining reliability, extensibility, and performance at scale.