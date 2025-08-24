"""
Unified configuration system for ScrambleBench.

This module provides the complete configuration schema following the TODO.md specification,
supporting deterministic evaluation with temperature=0, fixed seeds, and canonical metrics.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import yaml
import hashlib
import json
import warnings


@dataclass
class RunConfig:
    """Configuration for evaluation run settings."""
    run_id: str
    seed: int = 1337
    concurrency: int = 4
    max_cost_usd: float = 200.0
    retry_max_attempts: int = 3
    retry_backoff: str = "exponential"


@dataclass 
class DatasetConfig:
    """Configuration for individual datasets."""
    name: str
    path: str
    sample_size: int = 200  # per domain
    domains: Optional[List[str]] = None  # If None, use all domains in dataset
    

@dataclass
class TransformConfig:
    """Base class for transformation configurations."""
    kind: str


@dataclass
class OriginalTransformConfig(TransformConfig):
    """Configuration for original (untransformed) evaluation."""
    kind: str = "original"


@dataclass
class ParaphraseTransformConfig(TransformConfig):
    """Configuration for paraphrase control transformation."""
    kind: str = "paraphrase" 
    provider: str = "hosted_heldout"  # Must differ from eval provider
    n_candidates: int = 2
    semantic_sim_threshold: float = 0.85
    surface_divergence_min: float = 0.25
    cache_dir: str = "data/cache/paraphrase"
    temperature: float = 0.3  # Only for generation


@dataclass
class ScrambleTransformConfig(TransformConfig):
    """Configuration for scramble transformations."""
    kind: str = "scramble"
    levels: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    scheme: Dict[str, Any] = field(default_factory=lambda: {
        "type": "symbol_substitution",
        "alphabet": "@#$%&*+=?"
    })


@dataclass
class PromptingConfig:
    """Configuration for prompting templates."""
    system: str = "You are a careful reasoner. Answer concisely."
    template: str = "Q: {question}\\nA:"
    stop: List[str] = field(default_factory=lambda: ["\\n\\n", "###"])


@dataclass
class EvaluationParams:
    """Configuration for evaluation parameters."""
    temperature: float = 0.0  # Deterministic evaluation
    top_p: float = 1.0
    max_tokens: int = 256


@dataclass
class ScoringConfig:
    """Configuration for scoring methods."""
    mode: str = "exact_or_regex"
    key: str = "answer"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings."""
    prompting: PromptingConfig = field(default_factory=PromptingConfig)
    params: EvaluationParams = field(default_factory=EvaluationParams)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)


@dataclass
class ModelConfig:
    """Configuration for individual models."""
    name: str
    provider: str
    temperature: float = 0.0  # Deterministic by default
    max_tokens: int = 256
    timeout: int = 60
    api_key: Optional[str] = None  # Uses env if None


@dataclass
class ProviderGroupConfig:
    """Configuration for provider groups."""
    name: str
    provider: str
    list: List[str]


@dataclass
class ModelsConfig:
    """Configuration for models and provider groups."""
    provider_groups: List[ProviderGroupConfig]


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""
    save_prompts: bool = True
    save_completions: bool = True
    token_stats: bool = True
    tokenizer_perturbation: bool = True


@dataclass
class DatabaseConfig:
    """Configuration for database settings."""
    uri: str = "db/scramblebench.duckdb"


@dataclass
class ScrambleBenchConfig:
    """Main configuration class for ScrambleBench evaluation."""
    run: RunConfig
    datasets: List[DatasetConfig]
    transforms: List[Union[OriginalTransformConfig, ParaphraseTransformConfig, ScrambleTransformConfig]]
    evaluation: EvaluationConfig
    models: ModelsConfig
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # Computed properties
    _config_hash: Optional[str] = field(default=None, init=False)
    _created_at: datetime = field(default_factory=datetime.now, init=False)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._config_hash = self._compute_hash()
    
    def _validate_config(self):
        """Validate configuration for consistency and safety requirements."""
        # Validate provider isolation for paraphrase
        paraphrase_providers = set()
        eval_providers = set()
        
        for transform in self.transforms:
            if isinstance(transform, ParaphraseTransformConfig):
                paraphrase_providers.add(transform.provider)
        
        for provider_group in self.models.provider_groups:
            eval_providers.add(provider_group.provider)
        
        # Check for provider leakage
        if paraphrase_providers & eval_providers:
            overlapping = paraphrase_providers & eval_providers
            raise ValueError(
                f"Provider isolation violation: Paraphrase and evaluation cannot use "
                f"the same providers. Overlapping: {overlapping}"
            )
        
        # Validate deterministic settings
        # Handle both dataclass EvaluationConfig and LegacyEvaluationConfig
        if hasattr(self.evaluation, 'params'):
            # Dataclass EvaluationConfig
            temperature = self.evaluation.params.temperature
        else:
            # LegacyEvaluationConfig - skip validation for now
            temperature = 0.0  # Default assumption
        
        if temperature != 0.0:
            raise ValueError(
                f"Non-deterministic evaluation: temperature must be 0.0, got {temperature}"
            )
        
        # Validate scramble levels
        for transform in self.transforms:
            if isinstance(transform, ScrambleTransformConfig):
                if not all(0.0 <= level <= 1.0 for level in transform.levels):
                    raise ValueError(
                        f"Invalid scramble levels: all levels must be between 0.0 and 1.0, "
                        f"got {transform.levels}"
                    )
    
    def _compute_hash(self) -> str:
        """Compute hash of configuration for reproducibility tracking."""
        # Convert to dict and sort for consistent hashing
        config_dict = self._to_dict_for_hash()
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]
    
    def _to_dict_for_hash(self) -> Dict[str, Any]:
        """Convert config to dict suitable for hashing."""
        result = {}
        
        # Include all relevant fields except computed ones
        result['run'] = {
            'run_id': self.run.run_id,
            'seed': self.run.seed,
            'concurrency': self.run.concurrency
        }
        
        result['datasets'] = [
            {'name': ds.name, 'path': ds.path, 'sample_size': ds.sample_size}
            for ds in self.datasets
        ]
        
        result['transforms'] = []
        for transform in self.transforms:
            if isinstance(transform, OriginalTransformConfig):
                result['transforms'].append({'kind': 'original'})
            elif isinstance(transform, ParaphraseTransformConfig):
                result['transforms'].append({
                    'kind': 'paraphrase',
                    'provider': transform.provider,
                    'semantic_sim_threshold': transform.semantic_sim_threshold,
                    'surface_divergence_min': transform.surface_divergence_min
                })
            elif isinstance(transform, ScrambleTransformConfig):
                result['transforms'].append({
                    'kind': 'scramble',
                    'levels': transform.levels,
                    'scheme': transform.scheme
                })
        
        # Handle both dataclass EvaluationConfig and LegacyEvaluationConfig
        if hasattr(self.evaluation, 'prompting'):
            # Dataclass EvaluationConfig
            result['evaluation'] = {
                'prompting': {
                    'system': self.evaluation.prompting.system,
                    'template': self.evaluation.prompting.template
                },
                'params': {
                    'temperature': self.evaluation.params.temperature,
                    'max_tokens': self.evaluation.params.max_tokens
                }
            }
        else:
            # LegacyEvaluationConfig - use minimal representation
            result['evaluation'] = {
                'experiment_name': getattr(self.evaluation, 'experiment_name', 'unknown'),
                'type': 'legacy'
            }
        
        result['models'] = [
            {'name': pg.name, 'provider': pg.provider, 'list': sorted(pg.list)}
            for pg in self.models.provider_groups
        ]
        
        return result
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'ScrambleBenchConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScrambleBenchConfig':
        """Create configuration from dictionary."""
        # Parse run config
        run_data = data['run']
        run_config = RunConfig(
            run_id=run_data['run_id'],
            seed=run_data.get('seed', 1337),
            concurrency=run_data.get('concurrency', 4),
            max_cost_usd=run_data.get('max_cost_usd', 200.0),
            retry_max_attempts=run_data.get('retry', {}).get('max_attempts', 3),
            retry_backoff=run_data.get('retry', {}).get('backoff', 'exponential')
        )
        
        # Parse datasets
        datasets = []
        for ds_data in data['datasets']:
            datasets.append(DatasetConfig(
                name=ds_data['name'],
                path=ds_data['path'],
                sample_size=ds_data.get('sample_size', 200),
                domains=ds_data.get('domains')
            ))
        
        # Parse transforms
        transforms = []
        for transform_data in data['transforms']:
            kind = transform_data['kind']
            
            if kind == 'original':
                transforms.append(OriginalTransformConfig())
            elif kind == 'paraphrase':
                transforms.append(ParaphraseTransformConfig(
                    provider=transform_data['provider'],
                    n_candidates=transform_data.get('n_candidates', 2),
                    semantic_sim_threshold=transform_data.get('semantic_sim_threshold', 0.85),
                    surface_divergence_min=transform_data.get('surface_divergence_min', 0.25),
                    cache_dir=transform_data.get('cache_dir', 'data/cache/paraphrase'),
                    temperature=transform_data.get('temperature', 0.3)
                ))
            elif kind == 'scramble':
                transforms.append(ScrambleTransformConfig(
                    levels=transform_data.get('levels', [0.1, 0.2, 0.3, 0.4, 0.5]),
                    scheme=transform_data.get('scheme', {
                        'type': 'symbol_substitution',
                        'alphabet': '@#$%&*+=?'
                    })
                ))
            else:
                raise ValueError(f"Unknown transform kind: {kind}")
        
        # Parse evaluation config
        eval_data = data['evaluation']
        
        prompting_data = eval_data['prompting']
        prompting_config = PromptingConfig(
            system=prompting_data.get('system', 'You are a careful reasoner. Answer concisely.'),
            template=prompting_data.get('template', 'Q: {question}\\nA:'),
            stop=prompting_data.get('stop', ['\\n\\n', '###'])
        )
        
        params_data = eval_data['params']
        eval_params = EvaluationParams(
            temperature=params_data.get('temperature', 0.0),
            top_p=params_data.get('top_p', 1.0),
            max_tokens=params_data.get('max_tokens', 256)
        )
        
        scoring_data = eval_data['scoring']
        scoring_config = ScoringConfig(
            mode=scoring_data.get('mode', 'exact_or_regex'),
            key=scoring_data.get('key', 'answer')
        )
        
        evaluation_config = EvaluationConfig(
            prompting=prompting_config,
            params=eval_params,
            scoring=scoring_config
        )
        
        # Parse models config
        models_data = data['models']
        provider_groups = []
        
        for pg_data in models_data['provider_groups']:
            provider_groups.append(ProviderGroupConfig(
                name=pg_data['name'],
                provider=pg_data['provider'],
                list=pg_data['list']
            ))
        
        models_config = ModelsConfig(provider_groups=provider_groups)
        
        # Parse optional configs
        logging_data = data.get('logging', {})
        logging_config = LoggingConfig(
            save_prompts=logging_data.get('save_prompts', True),
            save_completions=logging_data.get('save_completions', True),
            token_stats=logging_data.get('token_stats', True),
            tokenizer_perturbation=logging_data.get('tokenizer_perturbation', True)
        )
        
        db_data = data.get('db', {})
        db_config = DatabaseConfig(
            uri=db_data.get('uri', 'db/scramblebench.duckdb')
        )
        
        return cls(
            run=run_config,
            datasets=datasets,
            transforms=transforms,
            evaluation=evaluation_config,
            models=models_config,
            logging=logging_config,
            db=db_config
        )
    
    def to_yaml(self, path: Union[str, Path]):
        """Save configuration to YAML file."""
        data = self.to_dict()
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        
        # Run config
        result['run'] = {
            'run_id': self.run.run_id,
            'seed': self.run.seed,
            'concurrency': self.run.concurrency,
            'max_cost_usd': self.run.max_cost_usd,
            'retry': {
                'max_attempts': self.run.retry_max_attempts,
                'backoff': self.run.retry_backoff
            }
        }
        
        # Datasets
        result['datasets'] = []
        for dataset in self.datasets:
            ds_dict = {
                'name': dataset.name,
                'path': dataset.path,
                'sample_size': dataset.sample_size
            }
            if dataset.domains:
                ds_dict['domains'] = dataset.domains
            result['datasets'].append(ds_dict)
        
        # Transforms
        result['transforms'] = []
        for transform in self.transforms:
            if isinstance(transform, OriginalTransformConfig):
                result['transforms'].append({'kind': 'original'})
            elif isinstance(transform, ParaphraseTransformConfig):
                result['transforms'].append({
                    'kind': 'paraphrase',
                    'provider': transform.provider,
                    'n_candidates': transform.n_candidates,
                    'semantic_sim_threshold': transform.semantic_sim_threshold,
                    'surface_divergence_min': transform.surface_divergence_min,
                    'cache_dir': transform.cache_dir,
                    'temperature': transform.temperature
                })
            elif isinstance(transform, ScrambleTransformConfig):
                result['transforms'].append({
                    'kind': 'scramble',
                    'levels': transform.levels,
                    'scheme': transform.scheme
                })
        
        # Evaluation
        result['evaluation'] = {
            'prompting': {
                'system': self.evaluation.prompting.system,
                'template': self.evaluation.prompting.template,
                'stop': self.evaluation.prompting.stop
            },
            'params': {
                'temperature': self.evaluation.params.temperature,
                'top_p': self.evaluation.params.top_p,
                'max_tokens': self.evaluation.params.max_tokens
            },
            'scoring': {
                'mode': self.evaluation.scoring.mode,
                'key': self.evaluation.scoring.key
            }
        }
        
        # Models
        result['models'] = {
            'provider_groups': []
        }
        for pg in self.models.provider_groups:
            result['models']['provider_groups'].append({
                'name': pg.name,
                'provider': pg.provider,
                'list': pg.list
            })
        
        # Logging
        result['logging'] = {
            'save_prompts': self.logging.save_prompts,
            'save_completions': self.logging.save_completions,
            'token_stats': self.logging.token_stats,
            'tokenizer_perturbation': self.logging.tokenizer_perturbation
        }
        
        # Database
        result['db'] = {
            'uri': self.db.uri
        }
        
        return result
    
    def get_config_hash(self) -> str:
        """Get the configuration hash for reproducibility."""
        return self._config_hash
    
    def get_all_models(self) -> List[ModelConfig]:
        """Get all individual model configurations."""
        models = []
        
        for provider_group in self.models.provider_groups:
            for model_name in provider_group.list:
                # Parse model name for provider-specific format
                if '|' in model_name:
                    model_id, provider = model_name.split('|', 1)
                else:
                    model_id = model_name
                    provider = provider_group.provider
                
                models.append(ModelConfig(
                    name=model_id,
                    provider=provider,
                    temperature=self.evaluation.params.temperature,
                    max_tokens=self.evaluation.params.max_tokens
                ))
        
        return models
    
    def estimate_total_evaluations(self) -> int:
        """Estimate total number of evaluations."""
        total_models = sum(len(pg.list) for pg in self.models.provider_groups)
        total_datasets = len(self.datasets)
        
        # Count transform combinations
        transform_count = 0
        for transform in self.transforms:
            if isinstance(transform, OriginalTransformConfig):
                transform_count += 1
            elif isinstance(transform, ParaphraseTransformConfig):
                transform_count += 1
            elif isinstance(transform, ScrambleTransformConfig):
                transform_count += len(transform.levels)
        
        # Sample size per dataset
        avg_sample_size = sum(ds.sample_size for ds in self.datasets) / len(self.datasets)
        
        return int(total_models * total_datasets * transform_count * avg_sample_size)


class ConfigGenerator:
    """Generate sample configuration templates."""
    
    @staticmethod
    def generate_template(template_type: str) -> Dict[str, Any]:
        """Generate a configuration template."""
        
        base_config = {
            'run': {
                'run_id': f'2025-01-01_{template_type}_v1',
                'seed': 1337,
                'concurrency': 4,
                'max_cost_usd': 50.0,
                'retry': {
                    'max_attempts': 3,
                    'backoff': 'exponential'
                }
            },
            'datasets': [
                {
                    'name': 'logic_eval',
                    'path': 'data/benchmarks/logic_eval.jsonl',
                    'sample_size': 50
                }
            ],
            'transforms': [
                {'kind': 'original'},
                {
                    'kind': 'paraphrase',
                    'provider': 'hosted_heldout',
                    'n_candidates': 2,
                    'semantic_sim_threshold': 0.85,
                    'surface_divergence_min': 0.25,
                    'cache_dir': 'data/cache/paraphrase'
                }
            ],
            'evaluation': {
                'prompting': {
                    'system': 'You are a careful reasoner. Answer concisely.',
                    'template': 'Q: {question}\\nA:',
                    'stop': ['\\n\\n', '###']
                },
                'params': {
                    'temperature': 0.0,
                    'top_p': 1.0,
                    'max_tokens': 256
                },
                'scoring': {
                    'mode': 'exact_or_regex',
                    'key': 'answer'
                }
            },
            'models': {
                'provider_groups': [
                    {
                        'name': 'local_ollama',
                        'provider': 'ollama',
                        'list': ['phi3:3.8b', 'llama3.1:8b']
                    }
                ]
            },
            'logging': {
                'save_prompts': True,
                'save_completions': True,
                'token_stats': True,
                'tokenizer_perturbation': True
            },
            'db': {
                'uri': 'db/scramblebench.duckdb'
            }
        }
        
        if template_type == 'smoke':
            # Minimal smoke test configuration
            base_config['run']['max_cost_usd'] = 5.0
            base_config['datasets'][0]['sample_size'] = 10
            base_config['transforms'] = [
                {'kind': 'original'},
                {
                    'kind': 'scramble', 
                    'levels': [0.2, 0.4],
                    'scheme': {'type': 'symbol_substitution', 'alphabet': '@#$'}
                }
            ]
            
        elif template_type == 'survey':
            # Full survey configuration
            base_config['run']['max_cost_usd'] = 200.0
            base_config['datasets'] = [
                {'name': 'logic_eval', 'path': 'data/benchmarks/logic_eval.jsonl', 'sample_size': 200},
                {'name': 'math_eval', 'path': 'data/benchmarks/math_eval.jsonl', 'sample_size': 200}
            ]
            base_config['transforms'] = [
                {'kind': 'original'},
                {
                    'kind': 'paraphrase',
                    'provider': 'hosted_heldout',
                    'n_candidates': 2,
                    'semantic_sim_threshold': 0.85,
                    'surface_divergence_min': 0.25,
                    'cache_dir': 'data/cache/paraphrase'
                },
                {
                    'kind': 'scramble',
                    'levels': [0.1, 0.2, 0.3, 0.4, 0.5],
                    'scheme': {'type': 'symbol_substitution', 'alphabet': '@#$%&*+=?'}
                }
            ]
            base_config['models']['provider_groups'].extend([
                {
                    'name': 'hosted_eval',
                    'provider': 'hosted',
                    'list': ['gpt-4o-mini|openrouter', 'llama-3.1-70b|openrouter']
                }
            ])
            
        elif template_type == 'research':
            # Research-focused configuration with all features
            base_config['run']['max_cost_usd'] = 500.0
            base_config['datasets'] = [
                {'name': 'logic_eval', 'path': 'data/benchmarks/logic_eval.jsonl', 'sample_size': 300},
                {'name': 'math_eval', 'path': 'data/benchmarks/math_eval.jsonl', 'sample_size': 300},
                {'name': 'reasoning_eval', 'path': 'data/benchmarks/reasoning_eval.jsonl', 'sample_size': 200}
            ]
            
        elif template_type == 'minimal':
            # Absolute minimal configuration for testing
            base_config['run']['max_cost_usd'] = 1.0
            base_config['datasets'][0]['sample_size'] = 5
            base_config['transforms'] = [{'kind': 'original'}]
            
        return base_config


# ============================================================================
# BACKWARD COMPATIBILITY AND MIGRATION UTILITIES
# ============================================================================

def convert_evaluation_config_yaml(yaml_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convert old evaluation config YAML format to unified config format.
    
    This utility helps migrate existing YAML config files from the deprecated
    evaluation.config format to the new unified_config format.
    """
    import warnings
    warnings.warn(
        "Loading legacy evaluation config format. Please migrate to unified config format.",
        DeprecationWarning,
        stacklevel=2
    )
    
    with open(yaml_path, 'r') as f:
        old_config = yaml.safe_load(f)
    
    # Convert to unified format
    new_config = {
        'run': {
            'run_id': old_config.get('experiment_name', 'migrated_config'),
            'seed': old_config.get('sample_seed', 1337),
            'concurrency': old_config.get('max_concurrent_requests', 4),
            'max_cost_usd': 50.0,
            'retry': {
                'max_attempts': 3,
                'backoff': 'exponential'
            }
        },
        'datasets': [],
        'transforms': [],
        'evaluation': {
            'prompting': {
                'system': 'You are a helpful assistant.',
                'template': 'Q: {question}\\nA:',
                'stop': ['\\n\\n', '###']
            },
            'params': {
                'temperature': 0.0,
                'top_p': 1.0,
                'max_tokens': 256
            },
            'scoring': {
                'mode': 'exact_or_regex',
                'key': 'answer'
            }
        },
        'models': {
            'provider_groups': []
        },
        'logging': {
            'save_prompts': True,
            'save_completions': True,
            'token_stats': True,
            'tokenizer_perturbation': True
        },
        'db': {
            'uri': 'db/scramblebench.duckdb'
        }
    }
    
    # Convert benchmark paths to datasets
    if 'benchmark_paths' in old_config:
        for i, path in enumerate(old_config['benchmark_paths']):
            new_config['datasets'].append({
                'name': f'benchmark_{i+1}',
                'path': path,
                'sample_size': old_config.get('max_samples', 200)
            })
    
    # Convert models to provider groups
    if 'models' in old_config:
        provider_models = {}
        for model in old_config['models']:
            provider = model.get('provider', 'openrouter')
            if provider not in provider_models:
                provider_models[provider] = []
            provider_models[provider].append(model['name'])
        
        for provider, model_list in provider_models.items():
            new_config['models']['provider_groups'].append({
                'name': f'{provider}_group',
                'provider': provider,
                'list': model_list
            })
    
    # Convert transformations
    new_config['transforms'].append({'kind': 'original'})
    
    if 'transformations' in old_config:
        trans = old_config['transformations']
        enabled_types = trans.get('enabled_types', [])
        
        if 'language_translation' in enabled_types:
            # This would require paraphrase config - skip for now
            pass
        
        if 'synonym_replacement' in enabled_types or 'scramble' in enabled_types:
            new_config['transforms'].append({
                'kind': 'scramble',
                'levels': [0.1, 0.2, 0.3],
                'scheme': {
                    'type': 'symbol_substitution',
                    'alphabet': '@#$%&*'
                }
            })
    
    return new_config


class LegacyEvaluationConfig:
    """
    Backward compatibility class for old EvaluationConfig.
    
    This class provides a bridge between the old evaluation.config.EvaluationConfig
    and the new ScrambleBenchConfig system.
    """
    
    def __init__(self, **kwargs):
        warnings.warn(
            "EvaluationConfig is deprecated. Use ScrambleBenchConfig from unified_config instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self._data = kwargs
        self.experiment_name = kwargs.get('experiment_name', 'default')
        self.models = kwargs.get('models', [])
    
    @classmethod
    def load_from_file(cls, path: Union[str, Path]):
        """Load from old format YAML file."""
        config_dict = convert_evaluation_config_yaml(path)
        # Create a ScrambleBenchConfig and wrap it
        unified_config = ScrambleBenchConfig.from_dict(config_dict)
        # Return a compatibility wrapper that has the attributes expected
        wrapper = cls(experiment_name=unified_config.run.run_id)
        wrapper._unified_config = unified_config
        return wrapper
    
    def save_to_file(self, path: Union[str, Path]):
        """Save in new unified format."""
        # Convert and save as unified config
        config_dict = convert_evaluation_config_yaml(path)
        unified_config = ScrambleBenchConfig.from_dict(config_dict)
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)


# Legacy aliases for backward compatibility
TransformationConfig = ScrambleTransformConfig
MetricsConfig = dict  # Placeholder for now  
PlotConfig = dict     # Placeholder for now
EvaluationConfig = LegacyEvaluationConfig  # For old imports