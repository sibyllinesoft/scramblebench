# ScrambleBench Documentation

This directory contains the comprehensive documentation for ScrambleBench using Sphinx.

## Documentation Structure

```
docs/
├── conf.py                 # Sphinx configuration
├── index.rst              # Main documentation index
├── Makefile               # Build automation (Unix)
├── make.bat               # Build automation (Windows)
├── _static/               # Static assets (CSS, images)
│   └── custom.css         # Custom styling
├── api/                   # API Reference Documentation
│   ├── core.rst          # Core framework API
│   ├── translation.rst   # Translation benchmarks API
│   ├── longcontext.rst   # Long context benchmarks API
│   ├── llm.rst           # LLM integration API
│   ├── evaluation.rst    # Evaluation pipeline API
│   ├── utils.rst         # Utilities API
│   └── cli.rst           # CLI reference
├── user_guide/           # User guides and tutorials
│   ├── installation.rst # Installation guide
│   ├── quickstart.rst   # Quick start guide
│   ├── cli_guide.rst    # CLI comprehensive guide
│   ├── configuration.rst # Configuration guide
│   └── evaluation_pipeline.rst # Evaluation system guide
├── tutorials/            # Step-by-step tutorials
│   ├── translation_benchmarks.rst
│   ├── long_context_benchmarks.rst
│   ├── custom_models.rst
│   ├── batch_evaluation.rst
│   └── configuration_examples.rst
├── examples/             # Code examples and notebooks
│   ├── basic_usage.rst
│   ├── advanced_usage.rst
│   ├── custom_integration.rst
│   └── notebooks/        # Jupyter notebooks
├── development/          # Development documentation
│   ├── contributing.rst
│   ├── testing.rst
│   └── architecture.rst
└── about/               # Project information
    ├── changelog.rst
    ├── license.rst
    └── citation.rst
```

## Building Documentation

### Prerequisites

Install documentation dependencies:

```bash
# Using uv (recommended)
uv sync --group docs

# Or using pip
pip install -e ".[docs]"
```

### Quick Build

```bash
# Build HTML documentation
cd docs/
make html

# Open documentation
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
start _build/html/index.html  # Windows
```

### Using Build Script

The project includes a comprehensive build script:

```bash
# Basic HTML build
scripts/build_docs.sh

# Clean build with verbose output
scripts/build_docs.sh -c -v

# Build and open in browser
scripts/build_docs.sh -o

# Auto-reload development server
scripts/build_docs.sh -w

# Build PDF documentation
scripts/build_docs.sh -t pdf

# Check for broken links
scripts/build_docs.sh -l
```

### Build Options

**Available formats:**
- `html` - HTML documentation (default)
- `pdf` - PDF documentation (requires LaTeX)
- `latex` - LaTeX source
- `epub` - EPUB e-book format

**Build script options:**
- `-h, --help` - Show help
- `-t, --type TYPE` - Build type (html, pdf, latex, epub)
- `-c, --clean` - Clean build directory
- `-v, --verbose` - Verbose output
- `-o, --open` - Open in browser after building
- `-l, --check-links` - Check for broken links
- `-w, --watch` - Auto-reload development server
- `-d, --output-dir DIR` - Custom output directory

## Development Workflow

### Local Development

1. **Start development server:**
   ```bash
   scripts/build_docs.sh -w
   ```
   Documentation will be available at http://localhost:8000

2. **Edit documentation files** - Changes will auto-reload

3. **Check for issues:**
   ```bash
   scripts/build_docs.sh -l  # Check links
   make doctest              # Run doctests
   ```

### Writing Documentation

#### RST (reStructuredText) Syntax

ScrambleBench documentation uses RST with Sphinx extensions:

```rst
Section Title
=============

Subsection
----------

**Bold text** and *italic text*

Code blocks:

.. code-block:: python

   from scramblebench import TranslationBenchmark
   
   benchmark = TranslationBenchmark(
       source_dataset="qa_data.json",
       language_type=LanguageType.SUBSTITUTION
   )

Links:
- External: `OpenRouter <https://openrouter.ai>`_
- Internal: :doc:`installation`
- API: :class:`scramblebench.core.benchmark.BaseBenchmark`

Lists:
* Item 1
* Item 2

Numbered lists:
1. First item
2. Second item

Admonitions:

.. note::
   This is a note

.. warning::
   This is a warning

.. code-block:: bash

   # Shell commands
   scramblebench --help
```

#### Docstring Standards

Use RST format in Python docstrings:

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """
    Short description of the function.
    
    Longer description with more details about what the function
    does and how to use it.
    
    :param param1: Description of first parameter
    :type param1: str
    :param param2: Description of second parameter with default
    :type param2: int
    :return: Description of return value
    :rtype: bool
    :raises ValueError: When param1 is empty
    
    Example:
        Basic usage::
        
            result = example_function("test", 20)
            if result:
                print("Success!")
    
    Note:
        Additional notes about usage or behavior.
    
    See Also:
        :func:`related_function`: Related functionality
    """
    pass
```

#### Cross-References

Link to other parts of documentation:

```rst
# Link to other documents
:doc:`installation` - Link to installation.rst
:doc:`../api/core` - Link to api/core.rst

# Link to Python objects
:class:`BaseBenchmark` - Link to class
:meth:`BaseBenchmark.run` - Link to method
:func:`scramblebench.utils.config.Config` - Link to function

# Link to sections
:ref:`configuration-files` - Link to section with label

# External links
`Python <https://python.org>`_ - External link
```

#### Code Examples

Include comprehensive code examples:

```rst
Complete Example:

.. code-block:: python

   from scramblebench import TranslationBenchmark
   from scramblebench.llm import OpenRouterClient
   from scramblebench.translation.language_generator import LanguageType
   
   # Create benchmark
   benchmark = TranslationBenchmark(
       source_dataset="qa_data.json",
       language_type=LanguageType.SUBSTITUTION,
       language_complexity=5
   )
   
   # Initialize model
   model = OpenRouterClient("openai/gpt-4")
   
   # Run evaluation
   result = benchmark.run(model, num_samples=100)
   print(f"Score: {result.score:.2%}")

Command line equivalent:

.. code-block:: bash

   scramblebench evaluate run \
     --models "openai/gpt-4" \
     --benchmarks "qa_data.json" \
     --experiment-name "test_run" \
     --max-samples 100
```

### Documentation Quality

#### Spell Check

```bash
# Install aspell (if needed)
sudo apt-get install aspell aspell-en  # Ubuntu/Debian
brew install aspell                    # macOS

# Check spelling
find docs/ -name "*.rst" -exec aspell check {} \;
```

#### Link Validation

```bash
# Check for broken links
scripts/build_docs.sh -l

# Manual link check
cd docs/
make linkcheck
```

#### Style Guide

1. **Consistency:**
   - Use consistent terminology
   - Follow existing formatting patterns
   - Maintain consistent code style in examples

2. **Clarity:**
   - Write clear, concise descriptions
   - Include practical examples
   - Explain complex concepts step-by-step

3. **Completeness:**
   - Document all public APIs
   - Include error conditions
   - Provide troubleshooting information

## Advanced Features

### API Documentation

Automatic API documentation generation:

```bash
# Generate API docs from source code
sphinx-apidoc -f -o docs/api_generated src/scramblebench

# Include in build
scripts/build_docs.sh
```

### Jupyter Notebooks

Include executable notebooks:

```rst
.. toctree::
   :maxdepth: 1
   
   notebooks/tutorial.ipynb
   notebooks/advanced_usage.ipynb
```

Requirements:
- Add notebooks to `docs/examples/notebooks/`
- Install `nbsphinx`: included in docs dependencies
- Notebooks are executed during build (unless configured otherwise)

### Mathematical Expressions

Include LaTeX math:

```rst
The accuracy is calculated as:

.. math::

   \text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}

Inline math: :math:`f(x) = x^2 + 2x + 1`
```

### Diagrams

Include Mermaid diagrams:

```rst
.. mermaid::

   graph TD
       A[Start] --> B{Decision}
       B -->|Yes| C[Action 1]
       B -->|No| D[Action 2]
       C --> E[End]
       D --> E
```

## Deployment

### GitHub Pages

Documentation can be deployed to GitHub Pages:

```bash
# Build documentation
scripts/build_docs.sh

# Deploy to gh-pages branch (example)
git checkout gh-pages
cp -r docs/_build/html/* .
git add .
git commit -m "Update documentation"
git push origin gh-pages
```

### Read the Docs

Configure for Read the Docs in `.readthedocs.yaml`:

```yaml
# .readthedocs.yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.11"

python:
  install:
    - requirements: docs/requirements.txt
    - method: pip
      path: .
      extra_requirements:
        - docs

sphinx:
  configuration: docs/conf.py
```

## Troubleshooting

### Common Issues

**Sphinx build errors:**
```bash
# Clean build
make clean
make html

# Check for syntax errors
python -m py_compile docs/conf.py
```

**Import errors in autodoc:**
```bash
# Check Python path
python -c "import scramblebench; print(scramblebench.__file__)"

# Install package in development mode
pip install -e .
```

**Missing dependencies:**
```bash
# Install all documentation dependencies
uv sync --group docs

# Or individual packages
pip install sphinx sphinx-rtd-theme myst-parser
```

**LaTeX/PDF errors:**
```bash
# Install LaTeX (Ubuntu/Debian)
sudo apt-get install texlive-latex-recommended texlive-fonts-recommended

# Install LaTeX (macOS)
brew install --cask mactex

# Build PDF
scripts/build_docs.sh -t pdf
```

### Performance Optimization

**Large documentation sets:**
- Use `sphinx-autobuild` for development
- Configure parallel building: `make html SPHINXOPTS="-j auto"`
- Disable extensions not needed for development builds

**Memory usage:**
- Configure autodoc to exclude large modules during development
- Use `autoapi` instead of `autodoc` for very large codebases

## Contributing

When contributing to documentation:

1. **Follow the style guide** outlined above
2. **Test your changes** with the build script
3. **Check for broken links** before submitting
4. **Include examples** for new features
5. **Update relevant cross-references**

For major documentation changes, consider:
- Creating an issue to discuss the approach
- Breaking large changes into smaller PRs
- Updating the documentation structure if needed

## References

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Sphinx RTD Theme](https://sphinx-rtd-theme.readthedocs.io/)
- [MyST Parser](https://myst-parser.readthedocs.io/) (Markdown support)