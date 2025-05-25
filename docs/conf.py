import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'AION'
author = 'Polymathic AI'

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
]

autosummary_generate = True

html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['style.css']

# Theme customizations to approximate Polymathic AI colors
html_theme_options = {
    'light_css_variables': {
        'color-brand-primary': '#6f42c1',
        'color-brand-content': '#6f42c1',
    },
    'dark_css_variables': {
        'color-brand-primary': '#a78bfa',
        'color-brand-content': '#a78bfa',
    },
}
