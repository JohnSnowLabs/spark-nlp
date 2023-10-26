# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../"))


# -- Project information -----------------------------------------------------

project = "Spark NLP"
copyright = "2023, John Snow Labs"
author = "John Snow Labs"

# The full version, including alpha/beta/rc tags
release = "5.1.3"
pyspark_version = "3.2.3"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.viewcode",
    # "sphinx.ext.autosectionlabel",
    # "sphinx.ext.autosummary",
    "numpydoc",  # handle NumPy documentation formatted docstrings.
    "sphinx-prompt",
    "sphinx_toggleprompt",
    # "sphinx_copybutton", # TODO
    "sphinx_substitution_extensions",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "autoapi.extension",
]

intersphinx_mapping = {
    "spark": ("https://spark.apache.org/docs/latest/api/python/", None),
}

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/_tf_graph_builders_1x/**",
    "**/_tf_graph_builders/**",
    "**/_templates/**",
    "**/_autoapi/**"
]


# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/logo.png"
html_favicon = "_static/fav.ico"

suppress_warnings = ["toc.excluded"]
add_module_names = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
# -- Options for autodoc --------------------------------------------------

# Look at the first line of the docstring for function and method signatures.
autodoc_docstring_signature = False
autosummary_generate = False
numpydoc_show_class_members = False  # Or add Method section in doc strings? https://stackoverflow.com/questions/65198998/sphinx-warning-autosummary-stub-file-not-found-for-the-methods-of-the-class-c
# autoclass_content = "both"  # use __init__ as doc as well

autoapi_options = [
    "members",
    "show-module-summary",
]
autoapi_type = "python"
autoapi_dirs = ["../sparknlp"]
autoapi_root = "reference/_autosummary"
autoapi_template_dir = "_templates/_autoapi"
autoapi_add_toctree_entry = False
# autoapi_member_order = "groupwise"
autoapi_keep_files = True
autoapi_ignore = exclude_patterns
# autoapi_generate_api_docs = False
# autoapi_python_use_implicit_namespaces = True

add_module_names = False

# -- More Configurations -----------------------------------------------------

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# Global substitutions in the RST files.
rst_prolog = """
.. |release| replace:: {0}
.. |pyspark_version| replace:: {1}
""".format(
    release, pyspark_version
)
