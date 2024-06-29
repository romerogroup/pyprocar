# Process to update pyprocar in pypi

1. Update version number in pyprocar/version.py
2. Update version number in setup.json
3. Update version number in README.md

4.  Recompile docs to update version.

    Only deletes the autosummarys directories
    ```bash
    make clean-autosummary && make html

    make deploy
    ```

    deletes the autosummarys, removes build directory directories
    ```bash
    make clean-except-examples && make html

    make deploy
    ```

    Complete clean, redos all the examples, remove builddir Takes the longest
    ```bash
    make clean && make html

    make deploy
    ```


5. Push these changes to the main branch.

6.  run this from the root directory `python setup.py sdist bdist_wheel`
7.  run this from the root directory `twine upload dist/*`
    - Username is `__token__`
    - Password is the apikey. You may need to create a new apikey on PyPi.

8. Update the releases on the main github page.