Contributing
============

We absolutely welcome contributions and we hope that this guide will
facilitate an understanding of the PyProcar code repository. It is
important to note that the PyProcar software package is maintained on a
volunteer basis and thus we need to foster a community that can support
user questions and develop new features to make this software a useful
tool for all users.

This page is dedicated to outline where you should start with your
question, concern, feature request, or desire to contribute.

Being Respectful
----------------

Please demonstrate empathy and kindness toward other people, other software,
and the communities who have worked diligently to build (un-)related tools.

Please do not talk down in Pull Requests, Issues, or otherwise in a way that
portrays other people or their works in a negative light.

Making contributions
-----------------------------

Contributions to PyProcar can be made by creating pull requests to the main branch. 
When you create a pull request, automated tests and documentation builds will be triggered 
to ensure code quality and documentation integrity. If these automated checks pass 
successfully, your pull request can be merged upon review by the maintainers.


## Walkthrough of the Contribution Process

### Forking the repository

First, you'll need to create your own fork of the PyProcar repository:

1. Navigate to `<https://github.com/romerogroup/pyprocar>`_
2. Click the "Fork" button in the top-right corner to create your own copy
3. Clone your forked repository (not the original) to your local machine:

```bash
git clone https://github.com/YOUR_USERNAME/pyprocar.git
cd pyprocar
```

4. Add the original repository as an upstream remote to keep your fork synchronized:

```bash
git remote add upstream https://github.com/romerogroup/pyprocar.git
```

5. Create a new branch for your contribution:

```bash
git checkout -b your-feature-branch-name
```

### Setting up the development environment

Next, we need to create a virtual environment and activate it.

#### Create a virtual environment

##### venv
```bash
python -m venv venv
```

```bash
# For linux
source venv/bin/activate

# For windows
venv\Scripts/activate.bat
```

##### conda
```bash
conda create -n pyprocar python=3.10
conda activate pyprocar
```

#### Installing dependencies
Next we need to install the dependencies for the project.

```bash
pip install -e .[dev]
```

### The pull request

Once you have made your changes, you're ready to create a pull request:

#### 1. Make your changes
Work on your feature branch and make the necessary code changes, ensuring you follow the project's coding standards and include appropriate documentation.

#### 2. Commit your changes
Create meaningful commit messages following the project's commit message guidelines:

```bash
git add .
git commit -m "feat: Add new feature description"
```

#### 3. Push your changes
Push your feature branch to your forked repository:

```bash
git push origin your-feature-branch-name
```

#### 4. Create the pull request
1. Navigate to your fork on GitHub
2. Click "Compare & pull request" button
3. Choose the `main` branch of the original repository as the base
4. Provide a clear title and description of your changes
5. Reference any related issues using `#issue-number`
6. Submit the pull request

#### 7. Review process
After submitting your pull request:
- Automated tests and documentation builds will run
- Maintainers will review your code
- You may need to make additional changes based on feedback
- Once approved and all checks pass, your pull request will be merged

#### 8. Keep your fork updated
After your pull request is merged, keep your fork synchronized:

```bash
git checkout main
git pull upstream main
git push origin main
```



Testing
---------

PyProcar uses pytest for testing to ensure code quality and functionality. This section will guide you through running tests and working with test data.

### Installing Test Data

Before running tests, you need to download the test data. PyProcar provides a script to automatically download all necessary test data:

```bash
python tests/scripts/download_test_data.py
```

This script will download all the required test data files to the appropriate directories within the test suite.

### Running Tests

Once you have the test data installed, you can run the full test suite using pytest:

```bash
pytest -v tests
```

The `-v` flag provides verbose output, showing detailed information about each test as it runs.

#### Running Specific Tests

You can also run specific test files or test functions:

```bash
# Run tests in a specific file
pytest -v tests/test_specific_module.py

# Run a specific test function
pytest -v tests/test_specific_module.py::test_function_name
```

#### Test Output

The tests will provide detailed output including:
- Which tests passed or failed
- Performance metrics
- Any warnings or errors encountered
- Coverage information if configured

### Contributing New Test Data

If you need to add new test data for your contributions:

1. **Contact the maintainers**: Email lllangWV@gmail.com to discuss uploading new test data
2. **Provide context**: Explain what the new test data represents and why it's needed
3. **Follow naming conventions**: Ensure your test data follows the project's naming and organization standards
4. **Update tests**: Create corresponding test cases that use your new test data

### Best Practices for Testing

When writing tests for your contributions:
- Write tests that cover both expected behavior and edge cases
- Use descriptive test function names that explain what is being tested
- Include docstrings in your test functions to explain complex test scenarios
- Ensure tests are independent and can run in any order
- Mock external dependencies when possible to keep tests fast and reliable



Documentation
---------------

PyProcar uses Sphinx to generate comprehensive documentation that is automatically built and hosted. This section explains how to work with and contribute to the documentation.

### Documentation Structure

The documentation source files are located in the `docs/source` directory. This includes:
- RST (reStructuredText) files for main documentation pages
- Configuration files for Sphinx
- Example notebooks and tutorials
- API reference documentation (auto-generated from docstrings)

### Automatic Building and Hosting

The documentation is hosted on Read the Docs, which provides automatic building:
- **Pull Requests**: Documentation builds are triggered automatically when you create a pull request, allowing reviewers to preview changes
- **Main Branch**: When changes are merged to the main branch, the live documentation is automatically updated
- **Multiple Versions**: Read the Docs maintains documentation for different versions of PyProcar

You can view the live documentation at the project's Read the Docs URL.

### Adding Examples

Examples are handled using nbsphinx, which allows Jupyter notebooks to be seamlessly integrated into the documentation:

#### Creating a New Example

1. **Create a Jupyter notebook**: Write your example in a `.ipynb` file with clear explanations and well-commented code
2. **Place the notebook**: Save it in the appropriate subdirectory within `docs/source/`
3. **Update the index**: Add your notebook to the relevant `index.rst` file to include it in the documentation structure

Example of adding to `index.rst`:
```rst
.. toctree::
   :maxdepth: 2
   
   existing_example
   your_new_example
```

#### Best Practices for Examples

- Include clear markdown cells explaining what each code section does
- Use realistic data and scenarios that users might encounter
- Keep examples focused on specific features or use cases
- Ensure all code cells run without errors
- Include relevant imports and setup code

### Docstring Style Guide

PyProcar uses the **numpydoc** style for docstrings. This provides consistent, readable documentation that integrates well with Sphinx.

#### Numpydoc Format Example

```python
def example_function(param1, param2, param3=None):
    """
    Brief description of what the function does.
    
    Longer description providing more details about the function's
    purpose, behavior, and any important considerations.
    
    Parameters
    ----------
    param1 : str
        Description of the first parameter.
    param2 : int or float
        Description of the second parameter.
    param3 : list, optional
        Description of the optional third parameter.
        Default is None.
    
    Returns
    -------
    result : numpy.ndarray
        Description of what is returned.
    
    Raises
    ------
    ValueError
        Description of when this exception is raised.
    
    Examples
    --------
    >>> result = example_function("test", 42)
    >>> print(result)
    array([1, 2, 3])
    
    Notes
    -----
    Any additional notes about the function's behavior,
    performance considerations, or related functions.
    
    References
    ----------
    .. [1] Reference to relevant paper or documentation
    """
    # Function implementation here
    pass
```

#### Key Sections in Numpydoc

- **Parameters**: Describe all function arguments with their types
- **Returns**: Detail what the function returns
- **Raises**: List exceptions that may be raised
- **Examples**: Provide usage examples (these will be tested if using doctest)
- **Notes**: Additional implementation details or usage notes
- **References**: Citations to papers, books, or other documentation

### Building Documentation Locally (Optional)

While Read the Docs handles automatic building, you can build documentation locally to preview changes:

```bash
cd docs
make clean
make html
```

The built documentation will be available in `docs/_build/html/`.

### Contributing to Documentation

When contributing documentation:
- Follow the existing structure and style
- Use clear, concise language
- Include examples where appropriate
- Test that notebooks run without errors
- Ensure all RST files are properly formatted
- Check that new content appears correctly in the table of contents
