"""Sphinx configuration for the HandUMI documentation."""

project = "HandUMI"
author = "HandUMI collaborators"
copyright = "2026, Robonet"
release = "0.1.0"
version = "0.1"

extensions = [
    "myst_parser",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

root_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
linkcheck_ignore = [r"http://localhost:\d+/?"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]
myst_heading_anchors = 3

html_theme = "sphinx_book_theme"
html_title = "HandUMI - Software"
html_logo = "_static/robonet-logo.svg"
html_favicon = "_static/favicon.svg"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_show_sphinx = False
html_last_updated_fmt = "%b %d, %Y"

html_theme_options = {
    "repository_url": "https://github.com/robonet-ai/handumi-sw",
    "repository_branch": "main",
    "path_to_docs": "docs/source",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "collapse_navigation": True,
    "use_sidenotes": True,
    "logo": {
        "text": "RoboNet",
        "image_light": "_static/robonet-logo.svg",
        "image_dark": "_static/robonet-logo-white.svg",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/robonet-ai/handumi-sw",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "HandUMI Hardware",
            "url": "https://github.com/BrikHMP18/HandUMI",
            "icon": "fa-solid fa-microchip",
            "type": "fontawesome",
        },
    ],
    "icon_links_label": "Quick links",
}
