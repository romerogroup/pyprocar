
___

# v6.5.0 (06-20-2025)

##### Bugs
- Fixed documentation link for Read the Docs
- Updated publication links in getting started documentation
- Updated license inclusion method in authors.rst
- Refactored reciprocal lattice vector extraction logic
- Increased max_distance in _keep_points_near_subset
- Handled empty Fermi surfaces in FermiDataHandler
- Updated download function import in __init__.py
- Renamed dir parameter to dirpath in extract_ideal_test_data

##### New Features
- Added GitHub workflow scripts for release management
- Revised README structure and added installation instructions
- Enhanced documentation with new showcase images and sections
- Added architecture overview and new image to documentation
- Enhanced download_test_data function with threading
- Improved tests and test data management
- Added Read the Docs configuration file for documentation build
- Enhanced merge message generation for pull requests
- Implemented commit message summarization with previous context
- Added PR summarization script and workflow

##### Documentation
- Updated documentation theme options and styling
- Revised contributing guidelines for clarity and completeness
- Added contributing guide to enhance community engagement
- Updated Sphinx configuration and added Monokai color style
- Updated commit message generation guidelines
- Added myst_parser to documentation dependencies

##### Maintenance
- Moved to Read the Docs; removed built docs and moved Sphinx source to docs directory
- Removed update_merge_commit.py script and associated workflow
- Organized test structure and updated dependencies
- Consolidated and reorganized import statements
- Cleaned up import statements and improved formatting
- Updated file handling to use Path objects in various classes

___

___

# v6.4.6 (06-07-2025)

##### Bugs
- None identified

##### New Features
- Added band indices validation in fermisurface

##### Documentation updates
- Updated CHANGELOG.md due to new release

##### Maintenance
- None identified

___

___

# v6.4.5 (05-27-2025)

##### Bugs
- Handled case of no rotations provided in ibz2fbz method

##### New features
- None identified

##### Documentation updates
- Updated CHANGELOG.md due to new release

##### Maintenance
- Merged branch 'main' of github.com:romerogroup/pyprocar into main

___

___

# v6.4.4 (05-25-2025)

##### Bugs
- Handled missing symmetry operations gracefully in the VASP module

##### New Features
- None identified

##### Documentation updates
- Updated CHANGELOG.md due to new release

##### Maintenance
- None identified

___

___

# v6.4.3 (04-30-2025)

##### Bugs
- Fixed issue in VASP where it returns an empty list for missing symmetry group operators

##### New features
- None identified

##### Documentation updates
- Updated CHANGELOG.md due to new release

##### Maintenance
- Merged branch 'main' of github.com:romerogroup/pyprocar into main

___

___

# v6.4.2 (04-29-2025)

##### Bugs
- None identified

##### New Features
- Added unit tests for VASP OUTCAR file parsing
- Added texture_clim parameter to FermiSurface3DConfig and updated FermiVisualizer for texture scaling
- Added properties to BrillouinZone2D for cell centers and face arrays
- Added FermiSurface class for 3D visualization of Fermi surfaces
- Added interpolation functions for 3D meshes using FFT
- Added weighted options to UnfoldingConfig and updated logging in scriptUnfold.py

##### Documentation
- Deleted updating_pyprocar.md
- Updated _version.py and CHANGELOG.md due to new release

##### Maintenance
- Updated workflow for releases
- Removed generated version file from repository
- Updated .gitignore to exclude pyprocar version file
- Updated .gitignore to include linked data directory
- Refactored various classes for improved readability, organization, and functionality
- Enhanced EBSPlot class with improved masking and logging
- Merged branch 'main' of github.com:romerogroup/pyprocar into main
- Updated .gitignore to include pyrightconfig.json and .cursor
- Removed logging size of color and width mask

___

___

# v6.4.1 (03-16-2025)

##### Bugs
- Removed logging statements in the Structure class due to issues with handling fractional coordinates being None.

##### New Features
- None identified

##### Documentation updates
- Updated _version.py and CHANGELOG.md for new release.

##### Maintenance
- Merged branch 'main' of github.com:romerogroup/pyprocar into main.

___

___

# v6.4.0 (03-16-2025)

##### Bugs
- Improved error handling and logging in parser and vasp modules. Update file path construction to use os.path.join for better compatibility. Improve exception handling to log specific errors when parsing VASP files.

##### New Features
- Enhance logging format in log_utils.py by including the function name in log messages for improved traceability during debugging.
- Update FermiSurface3DConfig and FermiSurface3D class by adjusting texture size and enhancing logging for better debugging.
- Enhance ElectronicBandStructure class by improving logging for reciprocal lattice and adding a new method for calculating the reciprocal space gradient of a scalar field using Fourier methods.
- Add baseline drawing functionality to DOSPlot and configuration options for baseline parameters in DensityOfStatesConfig.
- Add logging support in parser and vasp modules for improved debugging and traceability.
- Enhance FermiSurface3DConfig and FermiVisualizer by introducing new configuration options for displaying the Brillouin zone and axes.
- Enhance Bandstructure2DConfig and BandStructure2DVisualizer by adding new configuration options for clipping the Brillouin zone and saving animations.

##### Documentation
- Update version.
- Update README.md.
- Update _version.py and CHANGELOG.md due to new release.

##### Maintenance
- Refactor various classes and methods for improved logging, code clarity, maintainability, organization, and consistency.
- Standardize parameter formatting and enhance logging messages.
- Introduce verbosity control and cache management for data parsing in multiple modules.
- Refactor logging implementation across multiple modules to use a centralized logger instance.
- Merge branches and pull requests for better integration.

___

___

# v6.3.3 (03-01-2025)

##### Bugs
- Fix Fermi surface contour path extraction method

##### New Features
- None identified

##### Documentation updates
- Minor formatting improvements in docstrings and code structure
- Updated example scripts and notebooks
- Update _version.py and CHANGELOG.md due to new release

##### Maintenance
- Adjusted default colormap in fermi_surface_2d.yml
- Refactor data directory paths in examples to use os.path.join for improved readability and consistency. Updated multiple scripts to ensure compatibility with the new path structure.
- Minor code style adjustments in the Structure class
- Merge branch 'main' of github.com:romerogroup/pyprocar into main

___

___

# 6.3.2 (01-12-2025)

##### Bugs
- Fixed bug with `EBSPlot` due to handling of `labels` keyword, now defaults to a list with an empty string to avoid errors.

##### New Features
- Changed the source of `__version__` to `.version` for automatic updates on releases to GitHub and PyPI.

##### Documentation
- Documentation update.

##### Maintenance
- Updated publish workflow.
- Updated `_version.py` and `CHANGELOG.md` due to new release.
- Merge branch 'main' of github.com:romerogroup/pyprocar into main.

___

___

# 6.3.1 (10-11-2024)

##### Bugs
- Bug Fix: The method to export band structure did not account for `self.kpath` being `None` due to `atomicplot`.

##### New Features
- Added updated workflow script.
  
##### Documentation
- Updated CHANGELOG.md.

##### Maintenance
- Updated GitHub workflow scripts.
- Updated `_version.py` and CHANGELOG.md due to new release.

___

___

# 6.3.0 (10-03-2024)

##### Bugs
- Bug fix in filter for handling more than 10K kpoints in band filtering
- Fixed issue in fermi3d cross section slicer where arrow was not initialized in non-spin texture case
- Bug fix in fermisurface2d plot's output from skimage contour, requiring interpolation to map back to kmesh
- Bug fix in parsing of high symmetry points grid value
- Added exception handling for scenarios where no fermisurface is found, indicating the structure may not be metallic
- Bug in scriptfermi2d where symmetry operation did not apply in spin texture case
- Fixed bug in ebs_plot due to a typo in grid_linestyle

##### New Features
- Added publishing back into workflow and completed testing
- Introduced option to export bandsplot data
- Added more configurations for user control over plot parameters
- Added conda env.yml for dependency management
- Implemented GitHub action workflow for automated deployment to PYPI
- Added procar symmetry operations to ElectronicBandStructure
- Allowed QE and ELK to automatically shift by Fermi level
- Modified the QE parser to retrieve Fermi energy from scf.out

##### Documentation
- Updated _version.py and CHANGELOG.md for the new release
- Enhanced CHANGELOG with recent changes upon package release

##### Maintenance
- Removed dependencies from requirements.txt and migrated them to pyproject.toml
- Corrected the consistency of QE output results in angstrom to match VASP
- Made changes to ensure that the list of modes is sourced from cfg/dos.py
- Added option to normalize DOS by integral or max, requiring normalization mode specification in dosplot

___
___

## Old Changelog
v6.2.1 Jul 23rd, 2024 -- New symmetrization method, new fermi surface projection method, ebs and dos refactor,  <br />
v6.2.0 Jun 29th, 2024 -- Major bug fixes to 3d plotting and other minor bug fixes <br />
v6.1.10 Jun 09th, 2024 -- Bug fix to dos, qe parser changes, elk parser changes, fermisurface changes  <br />
v6.1.9 Mar 28th, 2024 -- Changed default fermi level behavior, updated dos implementation, and doc updates <br />
v6.1.8 Mar 5th, 2024 -- Bug fixes to stack_orbital in dosplot <br />
v6.1.7 Jan 15th, 2024 -- Bug fixes and doc updates <br />
v6.1.6 Oct 10th, 2023 -- Bandsplot, dosplot, bandsdosplot chnages, bug fixes, doc updates <br />
v6.1.5 Oct 10th, 2023 -- Feature additions, example gallery additions, and doc updates <br />
v6.1.4 Aug 18th, 2023 -- Bug fixes, example gallery additions, and doc updates <br />
v6.1.3 Aug 7th, 2023 -- Updated install requirements <br />
v6.1.2 Aug 7th, 2023 -- Bug fix and doc update <br />
v6.1.1 Aug 7th, 2023 -- Bug fix <br />
v6.1.0 Aug 7th, 2023 -- Bug fixes, doc update, config files <br />
v6.0.0 Jun 10th, 2023 -- Major code base changes. <br />
v5.6.6 Mar 6th, 2022 -- QE, bandsplot, dosplot, fermi surface, and band unfolding bug fixes. Directory change, parsers are now in the io directory. <br />
v5.6.5 Jun 10th, 2021 -- Fermi surface object and fermi surface plotter bug fixes <br />
v5.6.4 May 6th, 2021 -- Updates to Fermi surface plotter. <br />
v5.6.3 Mar 5th, 2021 -- QE and elk bug fixes. <br />
v5.6.2 Jan 11th, 2021 -- Updates and bugfixes to fermi surface and dos plotter. <br />
v5.6.1 Dec 7th, 2020 -- Fixed bug in PyProcar.cat() for merging parallel Abinit files for spin polarized calculations. Converted units Ha to eV. <br />
v5.6.0 Nov 30th, 2020 -- Repairs PROCAR file by default. Set flag repair=False to disable. <br />
v5.5.8 Nov 24th, 2020 -- Updates to parametric band structure plotting. Ability to change linewidths with ``linewidth`` flag. <br />
v5.4.4 Oct 23rd, 2020 -- Updates to DOS plotting, Fermi3D and bxsf parser and other bugfixes. <br />
v5.5.2 July 27th, 2020 -- Updated spin colinear calculations for Quantum Espresso and Lobster codes. <br />
v5.4.3 July 25th, 2020 -- Bug fixes in stand-alone version and updates to bandgap calculation. <br />
v5.4.0 Jun 17th, 2020 -- Improved 3D Fermi Surface plotter, added support for Quantum Espresso, conda support.  <br />
v5.3.3 May 22nd, 2020 -- Added DOS plotting feature. <br />
v5.2.1 May 11th, 2020 -- Bugfixes in pyprocar.cat and improving comparison method. <br />
v5.2.0 Apr 21st, 2020 -- Added spin colinear plotting feature for Elk calculations and a method to plot spin up and spin down plots separately without the need to filter the PROCAR file. <br />
v5.1.9 Apr 14th, 2020 -- Added feature to filter colinear spins in pyprocar.filter(). <br />
v5.1.8 Mar 27th, 2020 -- Fix iband reading error due to vasp incorrectly writting iband>999. <br />
v5.1.5 Mar 8th, 2020 -- Fixed summation issues in ElkParser. <br />
v5.1.4 Mar 7th, 2020 -- Added new class for parsing Abinit data.<br />
v5.1.3 Mar 5th, 2020 -- Fixed Abinit PROCAR formatting issues in PyProcar cat function.<br />
v5.1.1 Mar 5th, 2020 -- Removed bandscompare() due to redundancy with exportplt.<br />
v5.1.0 Mar 4th, 2020 -- Elk implementation.<br />
v5.0.1 Mar 2nd, 2020 -- Added orbital header array for newer version of VASP.<br />
v5.0.0 Mar 1st, 2020 -- Added discontinuous band-plotting feature and other improvements. <br />
v4.1.4 Feb 28th, 2020 -- Added option to convert k-points between reduced and cartesian when OUTCAR is supplied. <br />
v4.1.3 Feb 27th, 2020 -- Renormalize alpha values in band unfolder for values > 1. <br />
v4.1.2 Feb 24th, 2020 -- Bug fixes in band unfolder. <br />
v4.1.1 Feb 12th, 2020 -- Added feature to compare two parametric plots with colormaps in bandscompare.<br />
v4.1.0 Jan 10th, 2020 -- Added feature to export plots as matplotlib.pyplot objects for further processing through matplotlib options. <br />
v4.0.4 Dec 6th, 2019 -- Added command-line compatibility to standalone version and better Latex rendering.<br />
v4.0.1 Nov 17th, 2019 -- Added feature to filter k-points. <br />
v4.0.0 Nov 6th, 2019 -- Various bug fixes. Release of standalone version. Updated documentation.<br />
v3.9.2 Oct 4, 2019 -- Fixed bug in 2D Kmesh generator. <br />
v3.9.1 Sep 15, 2019 -- Fixed unfold spin polarized eigenvalue bug and spin up/down band energy error in unfolding.<br />
v3.9.0 Sep 12, 2019 -- Fixed spin polarized band unfolding.  <br />
v3.8.9 Sep 9, 2019 -- Added bbox_inches='tight' for savefig.<br />
v3.8.8 Jul 24, 2019 -- Fixed ambiguity in spin flag. <br />
v3.8.7 Jul 21, 2019 -- Fixed bug in K-mesh generator. <br />
v3.8.6 Jun 26, 2019 -- Bug fixes in band unfolding Fermi shift energy and band structure labels for Fermi shifts. <br />
v3.8.5 Jun 13, 2019 -- Bug fixes in Fermi surface plotting. <br />
v3.8.4 Jun 11, 2019 -- Fixed parsing old PROCAR format. <br />
v3.8.3 Jun 05, 2019 -- Updated parsing for PROCAR with phase. <br />
v3.8.2 Jun 05, 2019 -- Updated docs. <br />
v3.8.1 Jun 05, 2019 -- Updated reading from gzip for binary data. Increased parsing speed when phase factors are present. <br />
v3.71 Jun 05, 2019 -- More bug fixes. <br />
v3.7 Jun 04, 2019 -- Bug fixes for Fermi surface.<br />
v3.6 Jun 04, 2019 -- Added 3D Fermi surface utility.<br />
v3.5 May 22, 2019 -- added automatic high symmetry point labeling from KPOINTS file.<br />
v3.4 May 21, 2019 -- Bug fixes for plotting and added capability to plot meta-GGA. <br />
v3.3 Mar 19, 2019 -- Added band unfolder. <br />
v3.2 Nov 26, 2018 -- Moved project to romerogroup.<br />
v3.1 Sep 19, 2018 -- Minor bug fixes. <br />
v3.0 Sep 17, 2018 -- Added method to compare two PROCARs. Moved to Python3. <br />
v2.9 Jul 29,2018 -- Created PyProcar Mailing list.<br />
v2.8 May 23,2018 -- Fixed procar.cat()<br />
v2.7 May 18,2018 -- Fixed out-of-bounds error in k path generator.<br />
v2.6 May 18,2018 -- Fixed more issues with fermi2D<br />
v2.5 May 18.2018 -- Fixed issue with Vector<br />
v2.4 May 18,2018 -- Fixed minor issues with fermi2D and procarsymmetry<br />
v2.3 May 17,2018 -- Added k path generator.<br />
v2.2 May 14,2018 -- Updated documentation.<br />
v2.1 Apr 03,2018 -- Fixed issue with input arguments when using OUTCAR as an input <br />
v2.0 Mar 21,2018 -- Created PyProcar package version with added support to Abinit. <br />
v0.1.0, June 10, 2013 -- Initial release.<br />

___
