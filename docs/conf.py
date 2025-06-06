import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "AION-1"
author = "Polymathic AI"
html_title = "AION"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",  # For cards and grids
    "sphinxcontrib.mermaid",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

myst_heading_anchors = 3

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["style.css"]

# Theme customizations - separate light and dark themes
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#CA0E4C",
        "color-brand-content": "#CA0E4C",
        "color-foreground-primary": "#2c3e50",  # Dark text for light mode
        "color-foreground-secondary": "#546e7a",
        "color-foreground-muted": "#90a4ae",
        "color-foreground-border": "#e0e0e0",
        "color-background-primary": "#ffffff",  # White background for light mode
        "color-background-secondary": "#f5f5f5",
        "color-background-hover": "#fafafa",
        "color-background-border": "#e0e0e0",
        "color-sidebar-background": "#fafafa",
        "color-sidebar-background-border": "#e0e0e0",
        "color-sidebar-brand-text": "#2c3e50",
        "color-sidebar-caption-text": "#546e7a",
        "color-sidebar-link-text": "#2c3e50",
        "color-sidebar-link-text--top-level": "#2c3e50",
        "color-sidebar-search-background": "#ffffff",
        "color-sidebar-search-border": "#e0e0e0",
        "color-sidebar-search-foreground": "#2c3e50",
        "color-admonition-background": "#f5f5f5",
        "color-api-background": "#f5f5f5",
        "color-api-background-hover": "#eeeeee",
        "color-highlight-on-target": "rgba(202, 14, 76, 0.1)",
        "color-inline-code-background": "rgba(202, 14, 76, 0.08)",
        "color-inline-code-text": "#CA0E4C",
    },
    "dark_css_variables": {
        "color-brand-primary": "#CA0E4C",
        "color-brand-content": "#CA0E4C",
        "color-foreground-primary": "#e0e0e0",
        "color-foreground-secondary": "#b0b0b0",
        "color-foreground-muted": "#909090",
        "color-foreground-border": "#2a2a2a",
        "color-background-primary": "#0a0a0a",
        "color-background-secondary": "#171717",
        "color-background-hover": "#1a1a1a",
        "color-background-border": "#2a2a2a",
        "color-sidebar-background": "#0f0f0f",
        "color-sidebar-background-border": "#2a2a2a",
        "color-sidebar-brand-text": "#e0e0e0",
        "color-sidebar-caption-text": "#b0b0b0",
        "color-sidebar-link-text": "#cccccc",
        "color-sidebar-link-text--top-level": "#e0e0e0",
        "color-sidebar-search-background": "#1a1a1a",
        "color-sidebar-search-border": "#2a2a2a",
        "color-sidebar-search-foreground": "#e0e0e0",
        "color-admonition-background": "#1a1a1a",
        "color-api-background": "#1a1a1a",
        "color-api-background-hover": "#262626",
        "color-highlight-on-target": "rgba(202, 14, 76, 0.15)",
        "color-inline-code-background": "rgba(202, 14, 76, 0.15)",
        "color-inline-code-text": "#ff7a9a",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# Add custom footer
html_context = {
    "default_mode": "auto",  # Let the user's browser preference decide
}

# Customize source link text
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ""

# Add custom favicon if available
# html_favicon = "_static/favicon.ico"

# Set custom logo for the top left
# html_logo = "_static/polymathic_logo.png"
