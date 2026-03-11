# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "ThermoKourt"
copyright = "2026, Bart R.H. Geurten"
author = "Bart R.H. Geurten"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
]

# Napoleon settings — support Google and NumPy style docstrings
napoleon_google_docstrings = True
napoleon_numpy_docstrings = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Mock imports so Sphinx can process modules without their runtime deps
autodoc_mock_imports = [
    "matplotlib", "PIL", "numpy", "cv2", "torch", "torchvision",
    "h5py", "pandas", "sklearn", "yaml", "skimage", "scipy",
]

# MyST settings (Markdown support)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "ThermoKourt"
html_static_path = []
html_theme_options = {
    "source_repository": "https://github.com/zerotonin/thermokourt",
    "source_branch": "main",
    "source_directory": "docs/",
}
