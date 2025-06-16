# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import shutil
import sys
from distutils.sysconfig import get_python_lib
from pathlib import Path

from pyprocar import __version__

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "PyProcar"
copyright = "2020, Romero Group"
author = "Uthpala Herath"
version = __version__


sys.path.insert(0, os.path.abspath("."))

SRC_DIR = Path(__file__).parent
REPO_ROOT = SRC_DIR.parent.parent
SRC_EXAMPLES_PATH = SRC_DIR / "examples"
REPO_EXAMPLES_PATH = REPO_ROOT / "examples"
CONTRIBUTING_PATH = REPO_ROOT / "CONTRIBUTING.md"


print(f"REPO_ROOT: {REPO_ROOT}")
print(f"SRC_DIR: {SRC_DIR}")
print(f"SRC_EXAMPLES_PATH: {SRC_EXAMPLES_PATH}")


# Copy Repo Examples to docs source directory
if SRC_EXAMPLES_PATH.exists():
    shutil.rmtree(SRC_EXAMPLES_PATH)
shutil.copytree(REPO_EXAMPLES_PATH, SRC_EXAMPLES_PATH)

shutil.copy(CONTRIBUTING_PATH, SRC_DIR / "CONTRIBUTING.md")


if os.environ.get("READTHEDOCS") == "True":

    site_path = get_python_lib()
    ffmpeg_path = os.path.join(site_path, "imageio_ffmpeg", "binaries")
    print("########")
    print("good1")
    [ffmpeg_bin] = [
        file for file in os.listdir(ffmpeg_path) if file.startswith("ffmpeg-")
    ]
    print("########*****")
    print("good2")
    try:
        os.symlink(
            os.path.join(ffmpeg_path, ffmpeg_bin), os.path.join(ffmpeg_path, "ffmpeg")
        )
    except FileExistsError:
        print("File is already there!!!!!!!")
    else:
        print("file created :)")
    print("good3")
    os.environ["PATH"] += os.pathsep + ffmpeg_path
    print("good4")


# The full version, including alpha/beta/rc tags
release = __version__


# -- pyvista configuration ---------------------------------------------------
import pyvista

# Manage errors
pyvista.set_error_output_file("errors.txt")
# Ensure that offscreen rendering is used for docs generation
pyvista.OFF_SCREEN = True  # Not necessary - simply an insurance policy
# Preferred plotting style for documentation
pyvista.set_plot_theme("document")
pyvista.global_theme.window_size = [1024, 768]
pyvista.global_theme.font.size = 22
pyvista.global_theme.font.label_size = 22
pyvista.global_theme.font.title_size = 22
pyvista.global_theme.return_cpos = False
# pyvista.set_jupyter_backend(None)
# Save figures in specified directory
pyvista.FIGURE_PATH = os.path.join(os.path.abspath("./images/"), "auto-generated/")
if not os.path.exists(pyvista.FIGURE_PATH):
    os.makedirs(pyvista.FIGURE_PATH)

# necessary when building the sphinx gallery
pyvista.BUILDING_GALLERY = True
os.environ["PYVISTA_BUILDING_GALLERY"] = "true"


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinxcontrib.autoyaml",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "pyvista.ext.plot_directive",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    # "sphinx_gallery.gen_gallery",
    # "sphinxcontrib.pdfembed",
    "numpydoc",
    "sphinxcontrib.youtube",
    "sphinxcontrib.video",
    "sphinx_new_tab_link",
    "myst_parser",
    "nbsphinx",
]
nbsphinx_allow_errors = True
pygments_style = "sphinx"
# Used to set up examples sections
# sphinx_gallery_conf = {
#     # convert rst to md for ipynb
#     "pypandoc": True,
#     # path to your examples scripts
#     "examples_dirs": ["../../examples/"],
#     # path where to save gallery generated examples
#     "gallery_dirs": ["examples"],
#     "doc_module": "pyprocar",
#     "image_scrapers": ("pyvista", "matplotlib"),
# }

# Configuration for the Napoleon extension
napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False
napoleon_custom_sections = [
    ("My Custom Section", "params_style"),  # Custom section treated like parameters
    ("Plot Appearance", "params_style"),  # Custom section treated like parameters
    ("Surface Configuration", "params_style"),  # Custom section treated like parameters
    ("Spin Settings", "params_style"),  # Custom section treated like parameters
    ("Axes and Labels", "params_style"),  # Custom section treated like parameters
    (
        "Brillouin Zone Styling",
        "params_style",
    ),  # Custom section treated like parameters
    (
        "Advanced Configurations",
        "params_style",
    ),  # Custom section treated like parameters
    ("Isoslider Settings", "params_style"),  # Custom section treated like parameters
    ("Miscellaneous", "params_style"),  # Custom section treated like parameters
    "Methods",
]


# See https://numpydoc.readthedocs.io/en/latest/install.html
numpydoc_use_plots = True
numpydoc_show_class_members = False
numpydoc_xref_param_type = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]
# source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# import pydata_sphinx_theme


# https://pradyunsg.me/furo/customisation/logo/
html_title = f"PyProcar Docs: v{version}"
html_theme = "furo"

html_logo = os.path.join("media", "images", "PyProcar-logo.png")
html_static_path = []


# html_theme_path = ["_themes", ]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "source_repository": "https://github.com/romerogroup/pyprocar/",
    "source_branch": "main",
    "source_directory": "docs/source",
    "github_url": "https://github.com/romerogroup/pyprocar",
    "icon_links": [],
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/romerogroup/pyprocar",
            "html": "",
            "class": "fa-brands fa-solid fa-github fa-2x",
        },
        {
            "name": "The Paper",
            "url": "https://doi.org/10.1016/j.cpc.2019.107080",
            "icon": "fa fa-file-text fa-fw",
        },
        {
            "name": "Contributing",
            "url": "https://github.com/romerogroup/pyprocar/blob/main/CONTRIBUTING.rst",
            "icon": "fa fa-gavel fa-fw",
        },
    ],
}


html_context = {
    "logo_link": "index.html",  # Specify the link for the logo if needed
}

html_css_files = [
    "css/custom.css",
    "notebook.css",
    "_static/nbsphinx-gallery.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
]

html_js_files = ["js/custom.js"]
pygments_style = "sphinx"
pygments_dark_style = "monokai_colors.ManimMonokaiStyle"

if not os.path.exists("media/images"):
    os.makedirs("media/images")

if not os.path.exists("media/videos/480p30"):
    os.makedirs("media/videos/480p30")

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "PyProcardoc"

# -- Options for LaTeX output ------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "PyProcar.tex", "PyProcar Documentation", "Uthpala Herath", "manual"),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "pyprocar", "PyProcar Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "PyProcar",
        "PyProcar Documentation",
        author,
        "PyProcar",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# -- Extension configuration -------------------------------------------------

# skip building the osmnx example if osmnx is not installed
