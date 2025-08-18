Command Line Interface
======================

The CLI module provides comprehensive command-line tools for language management, batch processing, and evaluation workflows.

.. currentmodule:: scramblebench.cli

CLI Module
----------

.. automodule:: scramblebench.cli
   :members:
   :undoc-members:
   :show-inheritance:

Command Reference
-----------------

The ScrambleBench CLI is organized into several command groups:

Language Commands
~~~~~~~~~~~~~~~~~

**Generate Languages:**

.. code-block:: bash

   scramblebench language generate <name> --type <type> --complexity <level>

**List Languages:**

.. code-block:: bash

   scramblebench language list [--format json]

**Show Language Details:**

.. code-block:: bash

   scramblebench language show <name> [--show-rules] [--limit N]

**Delete Languages:**

.. code-block:: bash

   scramblebench language delete <name> [--force]

Transform Commands
~~~~~~~~~~~~~~~~~~

**Transform Text:**

.. code-block:: bash

   scramblebench transform text <text> <language>

**Replace Proper Nouns:**

.. code-block:: bash

   scramblebench transform proper-nouns <text> --strategy <strategy>

**Replace Synonyms:**

.. code-block:: bash

   scramblebench transform synonyms <text> --replacement-rate <rate>

Batch Commands
~~~~~~~~~~~~~~

**Extract Vocabulary:**

.. code-block:: bash

   scramblebench batch extract-vocab <file> --min-freq <freq> --max-words <count>

**Transform Datasets:**

.. code-block:: bash

   scramblebench batch transform <file> <language> --batch-size <size>

Evaluation Commands
~~~~~~~~~~~~~~~~~~~

**Run Evaluation:**

.. code-block:: bash

   scramblebench evaluate run --models <models> --benchmarks <files> --experiment-name <name>

**Analyze Results:**

.. code-block:: bash

   scramblebench evaluate analyze <experiment-name>

**Compare Experiments:**

.. code-block:: bash

   scramblebench evaluate compare <exp1> <exp2> [<exp3>...]

Utility Commands
~~~~~~~~~~~~~~~~

**Language Statistics:**

.. code-block:: bash

   scramblebench util stats <language>

**Export Rules:**

.. code-block:: bash

   scramblebench util export-rules <language> --format <format>

**Validate Transformation:**

.. code-block:: bash

   scramblebench util validate <language> <text>

Global Options
~~~~~~~~~~~~~~

All commands support these global options:

- ``--data-dir DIR``: Set data directory
- ``--config FILE``: Use configuration file
- ``--verbose``: Enable verbose output
- ``--quiet``: Suppress output
- ``--output-format FORMAT``: Set output format (text, json, yaml)
- ``--help``: Show help message