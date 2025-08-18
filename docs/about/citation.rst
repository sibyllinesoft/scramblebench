Citation
========

If you use ScrambleBench in your research, we kindly ask that you cite our work. This section provides detailed citation information and guidelines for academic and professional use.

Quick Citation
--------------

**For immediate use**, here's the recommended BibTeX citation:

.. code-block:: bibtex

   @software{scramblebench2024,
     title={ScrambleBench: Contamination-Resistant LLM Evaluation Through Constructed Languages},
     author={Rice, Nathan},
     year={2024},
     url={https://github.com/sibyllinesoft/scramblebench},
     version={0.1.0},
     doi={10.5281/zenodo.TODO}
   }

Academic Citation Formats
--------------------------

APA Style (7th Edition)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Rice, N. (2024). ScrambleBench: Contamination-resistant LLM evaluation through 
   constructed languages (Version 0.1.0) [Computer software]. 
   https://github.com/sibyllinesoft/scramblebench

MLA Style (9th Edition)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Rice, Nathan. "ScrambleBench: Contamination-Resistant LLM Evaluation Through 
   Constructed Languages." GitHub, 2024, 
   github.com/sibyllinesoft/scramblebench.

Chicago Style (17th Edition)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Author-Date System:**

.. code-block:: text

   Rice, Nathan. 2024. "ScrambleBench: Contamination-Resistant LLM Evaluation 
   Through Constructed Languages." Version 0.1.0. Computer software. 
   https://github.com/sibyllinesoft/scramblebench.

**Notes-Bibliography System:**

.. code-block:: text

   Nathan Rice, "ScrambleBench: Contamination-Resistant LLM Evaluation Through 
   Constructed Languages," version 0.1.0, computer software, accessed 
   December 18, 2024, https://github.com/sibyllinesoft/scramblebench.

IEEE Style
~~~~~~~~~~

.. code-block:: text

   N. Rice, "ScrambleBench: Contamination-resistant LLM evaluation through 
   constructed languages," 2024. [Online]. Available: 
   https://github.com/sibyllinesoft/scramblebench

Research Context and Methodology
---------------------------------

Background
~~~~~~~~~~

ScrambleBench addresses a critical problem in large language model evaluation: **training data contamination**. Traditional benchmarks may be inadvertently included in model training data, leading to inflated performance scores that don't reflect genuine reasoning capabilities.

Key Contributions
~~~~~~~~~~~~~~~~~

When citing ScrambleBench, you may want to reference these key methodological contributions:

1. **Constructed Language Framework**: A systematic approach to creating artificial languages that preserve logical structure while eliminating lexical overlap with training data.

2. **Document Transformation Pipeline**: Intelligent modification of long-context documents that maintains semantic content while changing surface form.

3. **Contamination Detection Methodology**: Empirical techniques for measuring the gap between memorization and genuine understanding in LLMs.

4. **Evaluation Metrics**: Novel metrics for assessing contamination resistance, including:
   - Coherence preservation scores
   - Entity relationship tracking
   - Position bias analysis
   - Failure pattern categorization

Methodological Framework
~~~~~~~~~~~~~~~~~~~~~~~~

ScrambleBench implements a **transformation-based evaluation paradigm** with the following principles:

- **Logical Preservation**: Transformations maintain the logical structure and solvability of problems
- **Lexical Elimination**: Surface forms are systematically altered to prevent memorization-based solutions
- **Verification Integrity**: Complete mappings enable verification of transformation accuracy
- **Statistical Rigor**: Results include confidence intervals and significance testing

Citation in Different Contexts
-------------------------------

Research Papers
~~~~~~~~~~~~~~~

**When introducing the contamination problem:**

.. code-block:: text

   Training data contamination poses a significant challenge to reliable LLM evaluation 
   (Rice, 2024). ScrambleBench addresses this through systematic transformation of 
   evaluation benchmarks into contamination-resistant variants.

**When describing methodology:**

.. code-block:: text

   We employed ScrambleBench (Rice, 2024) to generate constructed language versions 
   of our evaluation tasks, using phonetic transformation with complexity level 5 
   to ensure logical preservation while eliminating lexical overlap.

**When reporting results:**

.. code-block:: text

   Evaluation was conducted using ScrambleBench v0.1.0 (Rice, 2024), comparing 
   model performance on original and transformed benchmarks to quantify the 
   contribution of memorization versus reasoning.

Technical Documentation
~~~~~~~~~~~~~~~~~~~~~~~

**In methodology sections:**

.. code-block:: text

   Benchmark transformation was performed using the ScrambleBench toolkit 
   (Rice, 2024), which provides systematic methods for creating 
   contamination-resistant evaluation tasks while preserving logical structure.

**In implementation details:**

.. code-block:: text

   We utilized the ScrambleBench TranslationBenchmark class with substitution 
   language type and complexity level 7, following the framework described 
   in Rice (2024).

Blog Posts and Articles
~~~~~~~~~~~~~~~~~~~~~~~

**For general audience:**

.. code-block:: text

   Using ScrambleBench, a new toolkit for honest AI evaluation (Rice, 2024), 
   we discovered that model performance dropped significantly when memorization 
   was eliminated from the evaluation process.

**For technical audience:**

.. code-block:: text

   ScrambleBench (Rice, 2024) enables researchers to create contamination-resistant 
   benchmarks through constructed language transformation, revealing the true 
   reasoning capabilities of large language models.

Dataset Attribution Guidelines
------------------------------

When publishing results obtained with ScrambleBench, please include:

Required Attribution
~~~~~~~~~~~~~~~~~~~~

1. **ScrambleBench Version**: Always specify the version used (e.g., v0.1.0)
2. **Transformation Parameters**: Document the specific settings used
3. **Original Dataset Credits**: Acknowledge the original benchmark datasets
4. **Methodology Reference**: Cite the ScrambleBench paper/software

Example Attribution Block
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ## Evaluation Methodology
   
   Evaluations were conducted using ScrambleBench v0.1.0 (Rice, 2024), which 
   transforms existing benchmarks into contamination-resistant variants. We used 
   the following configuration:
   
   - Language Type: Phonetic substitution
   - Complexity Level: 5
   - Preservation Mode: Logical structure maintained
   - Original Datasets: GSM8K (Cobbe et al., 2021), LogiQA (Liu et al., 2020)
   
   Original dataset citations:
   - Cobbe, K., et al. (2021). Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.
   - Liu, J., et al. (2020). LogiQA: A challenge dataset for machine reading comprehension with logical reasoning. IJCAI.

Contributing Authors and Acknowledgments
-----------------------------------------

Core Development Team
~~~~~~~~~~~~~~~~~~~~~

**Nathan Rice** - Lead Developer and Research Architect
  - Framework design and implementation
  - Constructed language algorithms
  - Documentation and testing infrastructure
  - Email: nathan.alexander.rice@gmail.com
  - ORCID: 0000-0000-0000-0000 (Placeholder - will be updated)

Research Contributors
~~~~~~~~~~~~~~~~~~~~~

*Future contributors will be acknowledged here as the project grows.*

Acknowledgments
~~~~~~~~~~~~~~~

ScrambleBench builds upon extensive prior work in:

- **Contamination Detection**: Research by Sainz et al. (2023), Brown et al. (2020)
- **Constructed Languages**: Linguistic principles from constructed language research
- **Evaluation Methodology**: Best practices from the AI evaluation community
- **Open Source Ecosystem**: Transformers, PyTorch, and other foundational libraries

Institutional Support
~~~~~~~~~~~~~~~~~~~~~

*This section will be updated as institutional affiliations and support are established.*

Related Publications
--------------------

**Forthcoming Publications**
  - Detailed methodology paper (in preparation)
  - Empirical study of contamination across major LLMs (planned)
  - Best practices guide for contamination-resistant evaluation (planned)

**Relevant Prior Work**
  - Brown, T. et al. (2020). Language models are few-shot learners. NeurIPS.
  - Sainz, O. et al. (2023). Do large language models leak private information? arXiv preprint.
  - Wei, J. et al. (2022). Emergent abilities of large language models. TMLR.

Usage in Academic Papers
-------------------------

Guidelines for Authors
~~~~~~~~~~~~~~~~~~~~~~

When using ScrambleBench in academic research:

1. **Cite the Software**: Use the appropriate citation format for your venue
2. **Describe Methodology**: Explain the transformation approach and parameters
3. **Report Version**: Always specify the ScrambleBench version used
4. **Document Configuration**: Include configuration files or parameters
5. **Acknowledge Limitations**: Discuss any limitations or assumptions
6. **Share Results**: Consider contributing findings back to the community

Reproducibility Standards
~~~~~~~~~~~~~~~~~~~~~~~~~

To support reproducible research:

- **Configuration Files**: Include YAML configuration files used
- **Random Seeds**: Document random seeds for deterministic results
- **Environment**: Specify Python version and key dependencies
- **Hardware**: Note any relevant hardware specifications for performance-sensitive evaluations

Example Methods Section
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ## Evaluation Methodology
   
   To address potential training data contamination, we employed ScrambleBench v0.1.0 
   (Rice, 2024) to create contamination-resistant versions of our evaluation tasks. 
   ScrambleBench transforms benchmark problems using constructed languages that 
   preserve logical structure while eliminating lexical overlap with potential 
   training data.
   
   ### Transformation Configuration
   
   We used phonetic substitution transformation with the following parameters:
   - Language complexity: 5 (moderate)
   - Preservation mode: Complete logical structure
   - Verification: Bidirectional mapping validation
   - Quality control: Manual verification of 10% of samples
   
   ### Statistical Analysis
   
   We report both original and transformed benchmark scores, with the difference 
   attributed to memorization effects. Statistical significance was assessed using 
   paired t-tests with Bonferroni correction for multiple comparisons (α = 0.05).

Getting DOI and Persistent Identifiers
---------------------------------------

**Coming Soon**: We are working to establish persistent identifiers for ScrambleBench:

- **Zenodo DOI**: Planned for each major release
- **Software Heritage**: Archival of all releases
- **ORCID Integration**: Author identification and contribution tracking

Updates and Notifications
-------------------------

For citation updates and new publication guidelines:

- **GitHub Releases**: https://github.com/sibyllinesoft/scramblebench/releases
- **Documentation**: This page will be updated with new citation formats
- **Mailing List**: *Coming soon* - notification system for major updates

Questions about Citation
-------------------------

If you have questions about how to cite ScrambleBench or need help with specific citation formats:

- **GitHub Issues**: Open an issue with the "citation" label
- **Email**: Contact the development team directly
- **Discussions**: Use GitHub Discussions for community input

We're committed to supporting researchers in properly citing and attributing ScrambleBench in their work.