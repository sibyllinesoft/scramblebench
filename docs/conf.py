"""
Sphinx configuration file for ScrambleBench documentation.

This file configures Sphinx to generate comprehensive documentation for the
ScrambleBench library, including API documentation, user guides, and tutorials.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'ScrambleBench'
copyright = '2024, Nathan Rice'
author = 'Nathan Rice'
release = '0.1.0'
version = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.githubpages',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'myst_parser',
    'sphinxcontrib.mermaid',
    'nbsphinx',
]

# Source file parsers
source_suffix = {
    '.rst': None,
    '.md': 'myst',
    '.ipynb': 'nbsphinx',
}

# The master toctree document
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints',
    '.pytest_cache',
    '__pycache__',
]

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

html_theme_options = {
    # Basic theme configuration
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    
    # Navigation and layout
    'prev_next_buttons_location': 'both',  # Show navigation buttons at top and bottom
    'style_external_links': True,  # Add external link icons
    'vcs_pageview_mode': 'view',
    'collapse_navigation': False,  # Keep navigation expanded for better UX
    'sticky_navigation': True,
    'navigation_depth': 6,  # Allow deeper navigation nesting
    'includehidden': True,
    'titles_only': False,
    
    # Enhanced styling
    'style_nav_header_background': '#2c3e50',  # Custom primary color
    'flyout_display': 'attached',  # Better mobile experience
    
    # Additional customization
    'version_selector': True,
    'language_selector': False,
}

html_context = {
    "display_github": True,
    "github_user": "nathanrice",
    "github_repo": "scramblebench",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_title = 'ScrambleBench Documentation'
html_short_title = 'ScrambleBench'
html_logo = None  # Can be updated when logo is available
html_favicon = None  # Can be updated when favicon is available

# Enhanced meta tags for better SEO and social sharing
html_meta = {
    'description': 'ScrambleBench: A comprehensive contamination-resistant evaluation toolkit for Large Language Models',
    'keywords': 'LLM, evaluation, benchmark, contamination, machine learning, AI',
    'author': 'Nathan Rice',
    'robots': 'index, follow',
    'viewport': 'width=device-width, initial-scale=1.0',
}

# Enhanced HTML context for better GitHub integration
html_context.update({
    'github_url': 'https://github.com/sibyllinesoft/scramblebench',
    'edit_page': True,
    'source_url_prefix': 'https://github.com/sibyllinesoft/scramblebench/blob/main/',
    'edit_url_prefix': 'https://github.com/sibyllinesoft/scramblebench/edit/main/',
})

# Custom sidebar templates
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'donate.html',
    ]
}

# Additional static paths for custom assets
html_extra_path = [
    # Can be used for robots.txt, custom pages, etc.
]

# Show source links
html_show_sourcelink = True
html_show_sphinx = False  # Hide "Created with Sphinx" footer
html_show_copyright = True

# Additional CSS and JS files
html_css_files.extend([
    # Add more CSS files as needed
    # 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
])

html_js_files = [
    # Custom JavaScript for enhanced functionality
    'custom.js',
]

# Template paths
templates_path = ['_templates']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

latex_documents = [
    (master_doc, 'scramblebench.tex', 'ScrambleBench Documentation',
     'Nathan Rice', 'manual'),
]

# -- Options for manual page output ------------------------------------------
man_pages = [
    (master_doc, 'scramblebench', 'ScrambleBench Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (master_doc, 'ScrambleBench', 'ScrambleBench Documentation',
     author, 'ScrambleBench', 'Contamination-resistant LLM evaluation toolkit.',
     'Miscellaneous'),
]

# -- Options for Epub output -------------------------------------------------
epub_title = project
epub_exclude_files = ['search.html']

# -- Extension configuration -------------------------------------------------

# -- HTML output options -----------------------------------------------------
html_theme_options.update({
    # Additional ReadTheDocs theme options
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'both',
    'style_external_links': True,
    'vcs_pageview_mode': 'view',
    'style_nav_header_background': '#2c3e50',
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 6,
    'includehidden': True,
    'titles_only': False,
    'flyout_display': 'attached',
})

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True  # Enhanced for better examples
napoleon_use_admonition_for_notes = True     # Enhanced for better notes
napoleon_use_admonition_for_references = True # Enhanced for references
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True  # Enhanced type preprocessing
napoleon_type_aliases = {
    # Define common type aliases for better documentation
    'PathLike': 'Union[str, os.PathLike]',
    'ArrayLike': 'Union[np.ndarray, List, Tuple]',
}
napoleon_attr_annotations = True
napoleon_custom_sections = [
    'Returns',
    'Yields', 
    'Args',
    'Arguments',
    'Parameters',
    'Param',
    'Params',
    'Keyword Arguments',
    'Keyword Args',
    'Key Arguments',
    'Key Args',
    'Raises',
    'Except',
    'Exceptions',
    'Example',
    'Examples',
    'Usage',
    'Note',
    'Notes',
    'Warning',
    'Warnings',
    'See Also',
    'References',
    'Todo',
]

# -- Autodoc settings --------------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'groupwise',  # Group by type (methods, attributes, etc.)
    'special-members': '__init__, __call__, __enter__, __exit__',
    'undoc-members': True,
    'exclude-members': '__weakref__, __dict__, __module__',
    'show-inheritance': True,
    'ignore-module-all': False,
}

# Enhanced autodoc settings for better API documentation
autodoc_typehints = 'both'  # Show in signature and description
autodoc_typehints_description_target = 'documented_params'
autodoc_typehints_format = 'short'  # Use short form for type hints
autodoc_class_signature = 'mixed'
autodoc_inherit_docstrings = True
autodoc_preserve_defaults = True
autodoc_mock_imports = []

# Enhanced type hint handling
autodoc_type_aliases = {
    'PathLike': 'PathLike',
    'ArrayLike': 'ArrayLike',
    'DictLike': 'DictLike',
}

# Better docstring processing
autodoc_docstring_signature = True
autodoc_strip_signature_backslash = True

# -- Autosummary settings ----------------------------------------------------
autosummary_generate = True
autosummary_imported_members = True

# -- Intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'transformers': ('https://huggingface.co/docs/transformers/', None),
}

# -- Copy button configuration -----------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_exclude = '.linenos, .gp'  # Exclude line numbers and prompts
copybutton_copy_empty_lines = False
copybutton_line_continuation_character = "\\"
copybutton_here_doc_delimiter = "EOF"

# -- MyST settings -----------------------------------------------------------
myst_enable_extensions = [
    "deflist",
    "tasklist", 
    "colon_fence",
    "linkify",
    "substitution",
    "html_admonition",
    "smartquotes",
    "replacements",
    "strikethrough",
    "fieldlist",
    "attrs_inline",
    "attrs_block",
]

myst_heading_anchors = 4  # Generate anchors for h1-h4
myst_footnote_transition = True
myst_dmath_double_inline = True
myst_dmath_allow_labels = True
myst_dmath_allow_space = True
myst_dmath_allow_digits = True

# MyST substitutions for common replacements
myst_substitutions = {
    "version": version,
    "release": release,
    "project": project,
}

# -- nbsphinx settings -------------------------------------------------------
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True
nbsphinx_timeout = 60

# Enhanced setup function for custom extensions and styling
def setup(app):
    """Enhanced setup function for ScrambleBench documentation."""
    # Add custom CSS
    app.add_css_file('custom.css')
    
    # Add custom JavaScript if needed
    # app.add_js_file('custom.js')
    
    # Custom directive for enhanced code examples
    from docutils.parsers.rst import directives
    from docutils import nodes
    
    def visit_enhancement_node(self, node):
        """Visit function for enhancement nodes."""
        pass
    
    def depart_enhancement_node(self, node):
        """Depart function for enhancement nodes."""
        pass
    
    # Register custom nodes and directives
    app.add_node(
        nodes.container,
        html=(visit_enhancement_node, depart_enhancement_node)
    )
    
    # Enhanced configuration
    app.config.html_experimental_html5_writer = True
    
    return {
        'version': '0.1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }

# -- Additional Sphinx extensions configuration ------------------------------

# Sphinx autodoc type hints
set_type_checking_flag = True
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True
typehints_use_rtype = True

# Enhanced search configuration
html_search_language = 'en'
html_search_options = {
    'type': 'default',
    'teaser_length': 200,
    'highlight_in_results': True,
}

# Social media and sharing
html_use_opensearch = f'{project} Documentation'

# Enhanced error handling
suppress_warnings = [
    'image.nonlocal_uri',
    'ref.citation',
    'ref.footnote',
    'ref.option',
]

# Nitpicky mode for strict reference checking (commented out for initial setup)
# nitpicky = True
# nitpick_ignore = [
#     ('py:class', 'type'),
#     ('py:class', 'object'),
# ]