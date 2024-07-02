# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = PyProcar
SOURCEDIR     = ./source
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile


# Clean build files, autosummary files, and other generated files/directories.
clean:
	rm -rf $(BUILDDIR)/*
	rm -rf $(SOURCEDIR)/examples/
	rm -f errors.txt
	rm -f sphinx_warnings.txt
	rm -rf $(SOURCEDIR)/images/auto-generated
	rm -f $(SOURCEDIR)/getting-started/external_examples.rst
	find $(SOURCEDIR) -type d -name "_autosummary" -exec rm -rf {} +

# Clean similar files/directories as above, but excluding examples.
clean-except-examples:
	rm -rf $(BUILDDIR)/*
	rm -f errors.txt
	rm -f sphinx_warnings.txt
	rm -rf $(SOURCEDIR)/images/auto-generated
	rm -f $(SOURCEDIR)/getting-started/external_examples.rst
	find $(SOURCEDIR) -type d -name "_autosummary" -exec rm -rf {} +

# Remove only autosummary files.
clean-autosummary:
	find $(SOURCEDIR) -type d -name "_autosummary" -exec rm -rf {} +

# Build HTML docs in parallel using all available CPUs. 
# WARNING: this is a resource hog.
phtml:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(DEFAULTS) $(O) -j auto

deploy:
	@cp -a _build/html/. ../docs

github:
	@make html
	@cp -a _build/html/. ../docs

html:
    python generate_config_docs.py
    $(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	cp -r _build/html/. ../docs/
    rm -rf _build/html
	rm -rf _build/doctrees

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
