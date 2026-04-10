# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "TorchBayesian"

copyright = f"{datetime.now().year}, Raphael Brodeur"

author = "Raphael Brodeur"

release = "0.2.1"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",           # Auto-doc
    "sphinx.ext.autosummary",       # Creates summaries
    "sphinx_design",                # Design stuff
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

html_static_path = ['_static']
html_css_files = ["custom.css"]

html_theme_options = {
    # Logo on left
    "logo": {
        "text": "",
        "image_light": "_static/logo_wordmark_light.png",
        "image_dark": "_static/logo_wordmark_dark.png",
        "alt_text": "TorchBayesian Logo",                   # For accessibility and SEO
    },

    # More drop down
    "header_links_before_dropdown": 4,

    # GitHub icon on right
    "github_url": "https://github.com/torchbayesian/torchbayesian",

    # Cleaner footer
    "footer_start": [],
    "footer_center": ["copyright"],
    "footer_end": [],

    # Do not show previous and next page
    "show_prev_next": False,
}

# Do not show sphinx watermark
html_show_sphinx = False

# Do not show "Page source" option
html_show_sourcelink = False
