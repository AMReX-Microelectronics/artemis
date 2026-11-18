# Minimal Sphinx configuration for artemis-em.readthedocs.io.
#
# The ARTEMIS documentation is maintained at https://amrex-microelectronics.github.io/
# This Read the Docs build only publishes a single page that redirects there.

import os

project = 'ARTEMIS'
author = 'ARTEMIS collaboration'

root_doc = 'index'
extensions = []

# Build only index.rst: exclude every other file and directory in this folder
# (the old, out-of-date documentation pages).
_here = os.path.dirname(os.path.abspath(__file__))
exclude_patterns = [
    name for name in os.listdir(_here)
    if name not in ('index.rst', 'conf.py')
]

html_theme = 'alabaster'
html_baseurl = 'https://amrex-microelectronics.github.io/'
html_show_sourcelink = False
html_copy_source = False
