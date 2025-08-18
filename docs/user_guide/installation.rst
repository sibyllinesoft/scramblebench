Installation Guide
==================

This guide covers how to install ScrambleBench and its dependencies on different platforms.

System Requirements
-------------------

**Minimum Requirements:**

* Python 3.9 or higher
* 4GB RAM (8GB recommended for large evaluations)
* 1GB disk space for installation
* Internet connection for model API access

**Recommended Setup:**

* Python 3.11+ for best performance  
* 16GB RAM for processing large documents
* GPU support for local model inference (optional)

Installation Methods
--------------------

Option 1: Using uv (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`uv <https://github.com/astral-sh/uv>`_ is the fastest Python package manager and is recommended for ScrambleBench:

.. code-block:: bash

   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Clone the repository
   git clone https://github.com/nathanrice/scramblebench.git
   cd scramblebench

   # Install ScrambleBench and dependencies
   uv sync

   # Install with all optional dependencies
   uv sync --group all

   # Activate the virtual environment
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate  # On Windows

Option 2: Using pip
~~~~~~~~~~~~~~~~~~~

Standard installation with pip:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/nathanrice/scramblebench.git
   cd scramblebench

   # Create virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   # or
   venv\Scripts\activate  # On Windows

   # Install in development mode
   pip install -e .

   # Install with optional dependencies
   pip install -e ".[nlp,docs,dev]"

Option 3: From PyPI (Future)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once published to PyPI:

.. code-block:: bash

   pip install scramblebench

   # With optional dependencies
   pip install "scramblebench[nlp,docs]"

Optional Dependencies
---------------------

ScrambleBench includes several optional dependency groups:

**Development Dependencies:**

.. code-block:: bash

   # Install development tools
   uv sync --group dev
   # or with pip
   pip install -e ".[dev]"

Includes: pytest, black, ruff, mypy, pre-commit, jupyter

**Documentation Dependencies:**

.. code-block:: bash

   # Install documentation tools  
   uv sync --group docs
   # or with pip
   pip install -e ".[docs]"

Includes: sphinx, sphinx-rtd-theme, myst-parser, nbsphinx

**NLP Dependencies:**

.. code-block:: bash

   # Install NLP libraries
   uv sync --group nlp  
   # or with pip
   pip install -e ".[nlp]"

Includes: nltk, spacy for advanced text processing

**All Dependencies:**

.. code-block:: bash

   # Install everything
   uv sync --group all
   # or with pip
   pip install -e ".[dev,docs,nlp]"

API Keys Setup
--------------

ScrambleBench requires API keys for LLM providers:

OpenRouter API Key
~~~~~~~~~~~~~~~~~~

1. Sign up at `OpenRouter <https://openrouter.ai>`_
2. Generate an API key in your dashboard
3. Set the environment variable:

.. code-block:: bash

   export OPENROUTER_API_KEY="your_api_key_here"

   # Or add to your shell profile
   echo 'export OPENROUTER_API_KEY="your_api_key_here"' >> ~/.bashrc
   source ~/.bashrc

On Windows (PowerShell):

.. code-block:: powershell

   $env:OPENROUTER_API_KEY="your_api_key_here"

   # Or set permanently
   [Environment]::SetEnvironmentVariable("OPENROUTER_API_KEY", "your_api_key_here", "User")

Configuration File (Alternative)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also store API keys in a configuration file:

.. code-block:: yaml

   # config.yaml
   model:
     api_key: "your_api_key_here"
     default_provider: "openrouter"

.. code-block:: bash

   # Use configuration file
   scramblebench --config config.yaml evaluate run ...

Verification
------------

Verify your installation:

.. code-block:: bash

   # Check CLI is working
   scramblebench --help

   # Test basic functionality
   scramblebench language generate test_lang --type substitution --complexity 3

   # Check Python import
   python -c "import scramblebench; print('✓ ScrambleBench imported successfully')"

   # Test with OpenRouter (requires API key)
   python -c "
   from scramblebench.llm import OpenRouterClient
   client = OpenRouterClient('openai/gpt-3.5-turbo')
   print('✓ OpenRouter client created successfully')
   "

Development Setup
-----------------

For development and contributing:

.. code-block:: bash

   # Clone with development setup
   git clone https://github.com/nathanrice/scramblebench.git
   cd scramblebench

   # Install with development dependencies
   uv sync --group dev

   # Install pre-commit hooks
   pre-commit install

   # Run tests to verify setup
   pytest

   # Run linting
   ruff check src/
   black --check src/

   # Type checking
   mypy src/

Platform-Specific Notes
------------------------

macOS
~~~~~

.. code-block:: bash

   # Install Homebrew if needed
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

   # Install Python via Homebrew
   brew install python@3.11

   # Install uv
   brew install uv

   # Then follow standard installation

Linux (Ubuntu/Debian)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Update package list
   sudo apt update

   # Install Python and pip
   sudo apt install python3.11 python3.11-venv python3-pip

   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Then follow standard installation

Linux (CentOS/RHEL/Fedora)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install Python
   sudo dnf install python3.11 python3-pip  # Fedora
   # or
   sudo yum install python3.11 python3-pip  # CentOS/RHEL

   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Then follow standard installation

Windows
~~~~~~~

1. **Install Python:**
   Download Python 3.11+ from `python.org <https://www.python.org/downloads/>`_

2. **Install uv:**

   .. code-block:: powershell

      # Using PowerShell
      powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

3. **Follow standard installation**

Docker Installation
-------------------

Use the provided Dockerfile for containerized deployment:

.. code-block:: bash

   # Build Docker image
   docker build -t scramblebench .

   # Run with environment variables
   docker run -e OPENROUTER_API_KEY="your_key" scramblebench \
     scramblebench evaluate run --models "gpt-3.5-turbo" \
     --benchmarks "data/benchmarks/simple_qa.json"

   # Mount local data directory
   docker run -v $(pwd)/data:/app/data \
     -e OPENROUTER_API_KEY="your_key" \
     scramblebench \
     scramblebench language generate test --type substitution

Common Issues
-------------

**ImportError: No module named 'scramblebench'**

Solution:

.. code-block:: bash

   # Ensure you're in the right directory and environment
   cd /path/to/scramblebench
   source .venv/bin/activate  # or activate your virtual environment
   pip install -e .

**OpenRouter API errors**

Solution:

.. code-block:: bash

   # Verify API key is set
   echo $OPENROUTER_API_KEY

   # Test API key manually
   curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     https://openrouter.ai/api/v1/models

**Memory issues with large evaluations**

Solution:

.. code-block:: yaml

   # Add to config.yaml
   evaluation:
     batch_size: 10  # Reduce batch size
     max_samples: 100  # Limit samples

   model:
     timeout: 30  # Reduce timeout
     rate_limit: 5.0  # Lower rate limit

**Permission errors on macOS/Linux**

Solution:

.. code-block:: bash

   # Fix permissions
   chmod +x scripts/*
   sudo chown -R $USER:$USER ~/.cache/scramblebench

Next Steps
----------

After installation:

1. **Quick Start:** See :doc:`quickstart` for basic usage
2. **CLI Guide:** Check :doc:`cli_guide` for command-line usage  
3. **Configuration:** Review :doc:`configuration` for advanced setup
4. **Examples:** Explore the ``examples/`` directory

Getting Help
------------

If you encounter issues:

1. **Check the logs:** Enable verbose mode with ``--verbose``
2. **GitHub Issues:** `Report issues <https://github.com/nathanrice/scramblebench/issues>`_
3. **Discussions:** `Ask questions <https://github.com/nathanrice/scramblebench/discussions>`_
4. **Documentation:** Browse this documentation for solutions