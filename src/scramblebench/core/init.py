"""
Project initialization utilities for ScrambleBench.

Creates new projects with sample configurations and directory structures
for different evaluation scenarios (minimal, survey, research).
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


class ProjectInitializer:
    """Initializes new ScrambleBench projects with templates."""
    
    def __init__(self):
        self.templates = {
            'minimal': self._minimal_template,
            'survey': self._survey_template,
            'research': self._research_template
        }
    
    def create_project(self, project_name: str, template: str = 'survey') -> Dict[str, Any]:
        """Create a new ScrambleBench project."""
        if template not in self.templates:
            raise ValueError(f"Unknown template: {template}")
        
        project_path = Path(project_name)
        project_path.mkdir(exist_ok=True)
        
        # Create directory structure
        directories = [
            "configs",
            "data/datasets",
            "data/cache",
            "results",
            "db"
        ]
        
        for directory in directories:
            (project_path / directory).mkdir(parents=True, exist_ok=True)
        
        # Create configuration files
        template_func = self.templates[template]
        configs = template_func()
        
        config_files = []
        for config_name, config_content in configs.items():
            config_path = project_path / "configs" / f"{config_name}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config_content, f, default_flow_style=False, indent=2)
            config_files.append(str(config_path))
        
        # Create sample dataset
        sample_dataset = self._create_sample_dataset()
        dataset_path = project_path / "data" / "datasets" / "sample.jsonl"
        with open(dataset_path, 'w') as f:
            for item in sample_dataset:
                f.write(json.dumps(item) + '\n')
        
        # Create README
        readme_content = self._create_readme(project_name, template)
        readme_path = project_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        return {
            'project_path': str(project_path),
            'template': template,
            'configs_created': config_files,
            'sample_dataset': str(dataset_path),
            'readme': str(readme_path)
        }
    
    def _minimal_template(self) -> Dict[str, Dict[str, Any]]:
        """Create minimal configuration template."""
        return {
            'smoke': {
                'run': {
                    'seed': 1337,
                    'max_concurrency': 2,
                    'max_tokens': 100,
                    'models': [
                        {
                            'provider': 'ollama',
                            'model_id': 'llama2:7b',
                            'provider_config': {
                                'base_url': 'http://localhost:11434'
                            }
                        }
                    ]
                },
                'datasets': [
                    {
                        'name': 'sample',
                        'path': 'data/datasets/sample.jsonl'
                    }
                ],
                'transforms': [
                    {
                        'type': 'original'
                    },
                    {
                        'type': 'scramble',
                        'scheme': {
                            'type': 'symbol_substitution',
                            'alphabet': '@#$%&*+=?'
                        },
                        'levels': [0.1, 0.3]
                    }
                ]
            }
        }
    
    def _survey_template(self) -> Dict[str, Dict[str, Any]]:
        """Create survey configuration template."""
        return {
            'smoke': self._minimal_template()['smoke'],
            'survey': {
                'run': {
                    'seed': 1337,
                    'max_concurrency': 4,
                    'max_tokens': 512,
                    'models': [
                        {
                            'provider': 'ollama',
                            'model_id': 'llama2:7b',
                            'provider_config': {
                                'base_url': 'http://localhost:11434'
                            }
                        },
                        {
                            'provider': 'ollama',
                            'model_id': 'gemma:7b',
                            'provider_config': {
                                'base_url': 'http://localhost:11434'
                            }
                        }
                    ]
                },
                'datasets': [
                    {
                        'name': 'sample',
                        'path': 'data/datasets/sample.jsonl'
                    }
                ],
                'transforms': [
                    {
                        'type': 'original'
                    },
                    {
                        'type': 'paraphrase',
                        'provider': 'llama2:7b',  # Different from eval models
                        'n_candidates': 3,
                        'semantic_sim_threshold': 0.85,
                        'surface_divergence_min': 0.25,
                        'cache_dir': 'data/cache/paraphrase'
                    },
                    {
                        'type': 'scramble',
                        'scheme': {
                            'type': 'symbol_substitution',
                            'alphabet': '@#$%&*+=?'
                        },
                        'levels': [0.1, 0.2, 0.3, 0.4, 0.5]
                    }
                ]
            }
        }
    
    def _research_template(self) -> Dict[str, Dict[str, Any]]:
        """Create research configuration template."""
        survey_configs = self._survey_template()
        
        # Add comprehensive research configuration
        survey_configs['research'] = {
            'run': {
                'seed': 1337,
                'max_concurrency': 8,
                'max_tokens': 1024,
                'models': [
                    # Ollama models
                    {
                        'provider': 'ollama',
                        'model_id': 'llama2:7b',
                        'provider_config': {'base_url': 'http://localhost:11434'}
                    },
                    {
                        'provider': 'ollama', 
                        'model_id': 'llama2:13b',
                        'provider_config': {'base_url': 'http://localhost:11434'}
                    },
                    {
                        'provider': 'ollama',
                        'model_id': 'gemma:7b', 
                        'provider_config': {'base_url': 'http://localhost:11434'}
                    },
                    # Hosted models (requires API keys)
                    {
                        'provider': 'openai',
                        'model_id': 'gpt-3.5-turbo',
                        'provider_config': {
                            'api_key': '${OPENAI_API_KEY}'
                        }
                    }
                ]
            },
            'datasets': [
                {
                    'name': 'sample',
                    'path': 'data/datasets/sample.jsonl'
                }
            ],
            'transforms': [
                {
                    'type': 'original'
                },
                {
                    'type': 'paraphrase',
                    'provider': 'llama2:13b',  # Separate paraphrase model
                    'n_candidates': 5,
                    'semantic_sim_threshold': 0.85,
                    'surface_divergence_min': 0.25,
                    'temperature': 0.3,
                    'cache_dir': 'data/cache/paraphrase'
                },
                {
                    'type': 'scramble',
                    'scheme': {
                        'type': 'symbol_substitution',
                        'alphabet': '@#$%&*+=?'
                    },
                    'levels': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
                }
            ]
        }
        
        return survey_configs
    
    def _create_sample_dataset(self) -> List[Dict[str, Any]]:
        """Create sample dataset for testing."""
        return [
            {
                "text": "What is the capital of France?",
                "answer": "Paris",
                "domain": "geography"
            },
            {
                "text": "Calculate 15 + 27.",
                "answer": "42", 
                "domain": "arithmetic"
            },
            {
                "text": "Who wrote the novel '1984'?",
                "answer": "George Orwell",
                "domain": "literature"
            },
            {
                "text": "What is the chemical symbol for gold?",
                "answer": "Au",
                "domain": "chemistry"
            },
            {
                "text": "In which year did World War II end?", 
                "answer": "1945",
                "domain": "history"
            },
            {
                "text": "What is the largest planet in our solar system?",
                "answer": "Jupiter",
                "domain": "astronomy"
            },
            {
                "text": "Who painted the Mona Lisa?",
                "answer": "Leonardo da Vinci", 
                "domain": "art"
            },
            {
                "text": "What is the square root of 64?",
                "answer": "8",
                "domain": "arithmetic"
            },
            {
                "text": "Which element has the atomic number 1?",
                "answer": "Hydrogen",
                "domain": "chemistry"
            },
            {
                "text": "What is the smallest country in the world?",
                "answer": "Vatican City",
                "domain": "geography"
            }
        ]
    
    def _create_readme(self, project_name: str, template: str) -> str:
        """Create README for the project."""
        return f"""# {project_name}

ScrambleBench project initialized with **{template}** template on {datetime.now().strftime('%Y-%m-%d')}.

## Getting Started

1. **Check available models:**
   ```bash
   scramblebench models list --test-connectivity
   ```

2. **Run a smoke test:**
   ```bash
   scramblebench evaluate run --config configs/smoke.yaml
   ```

3. **View results:**
   ```bash
   scramblebench evaluate status
   ```

## Configuration Files

- `configs/smoke.yaml` - Quick test with minimal models/transforms
- `configs/survey.yaml` - Full survey evaluation (if using survey/research template)
- `configs/research.yaml` - Comprehensive research setup (if using research template)

## Dataset

- `data/datasets/sample.jsonl` - Sample 10-item test dataset

## Project Structure

```
{project_name}/
├── configs/           # YAML configuration files
├── data/
│   ├── datasets/      # Dataset files (.jsonl)
│   └── cache/         # Paraphrase cache
├── results/           # Analysis outputs
├── db/               # DuckDB database files
└── README.md         # This file
```

## Next Steps

1. **Add your datasets** to `data/datasets/`
2. **Configure models** in the YAML configs (add API keys for hosted models)
3. **Customize transforms** - adjust scramble levels, paraphrase settings
4. **Run evaluations** and analyze results with statistical models

## Documentation

- Run `scramblebench --help` for command reference
- See individual command help: `scramblebench evaluate --help`
- Check the ScrambleBench documentation for detailed configuration options

## Template: {template.title()}

{self._get_template_description(template)}
"""

    def _get_template_description(self, template: str) -> str:
        """Get description for template type."""
        descriptions = {
            'minimal': """
**Minimal Template**
- Single Ollama model (llama2:7b)
- Original + basic scramble transforms
- 10-item sample dataset
- Perfect for quick testing and development
            """.strip(),
            'survey': """
**Survey Template** 
- Multiple Ollama models for comparison
- All transform types: original, paraphrase, scramble
- Designed for comprehensive model surveys
- Includes paraphrase generation with provider isolation
            """.strip(),
            'research': """
**Research Template**
- Comprehensive model set (Ollama + hosted APIs)
- Full statistical analysis pipeline
- Advanced paraphrase generation settings
- Publication-ready configuration
- Includes analysis and export commands
            """.strip()
        }
        return descriptions.get(template, "")


class ConfigGenerator:
    """Generate configuration templates."""
    
    def generate_template(self, template_type: str) -> Dict[str, Any]:
        """Generate configuration template."""
        initializer = ProjectInitializer()
        
        if template_type == 'smoke':
            return initializer._minimal_template()['smoke']
        elif template_type == 'survey':
            return initializer._survey_template()['survey']
        elif template_type == 'research':
            return initializer._research_template()['research']
        elif template_type == 'minimal':
            return initializer._minimal_template()['smoke']
        else:
            raise ValueError(f"Unknown template type: {template_type}")