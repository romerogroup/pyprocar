
___

# 6.3.0 (10-03-2024)

##### Bugs
- Bug fix in filter; was not properly filtering bands with more than 10K kpoints.
- Bug in fermi3d cross section slicer; arrow not initialized in non-spin texture case.
- Bug fix in fermisurface2d plot; skimage contour output was using mesh index points instead of kmesh grid, requiring interpolation.
- Bug fix in parsing of high symmetry points grid value.
- Bug in scriptfermi2d; the symmetry operation did not apply in spin texture case.
- Fix ebs_plot bug due to a typo of grid_linestyle.

##### New Features
- Added conda env.yml.
- Added GitHub action workflow to automate deployment on package release; includes building and deploying to PYPI, prepending recent changes in CHANGELOG, updating the repo version, and updating the release notes.
- Added procar symmetry operations to ElectronicBandStructure.
- Added option to export bandsplot data.
- Added more configurations for user control of plots via *_params for various matplotlib functions.
- Switched behavior to allow QE and ELK to automatically shift by Fermi.
- Changed QE parser to get Fermi energy from scf.out; in nscf, the .xml gets overwritten, but in bands, the value does not.
- Added option to normalize DOS by integral or max, requiring specification of normalize_dos_mode='max' or 'integral' in dosplot.

##### Documentation
- Removed dependencies from requirements.txt, moved them to pyproject.toml.

##### Maintenance
- Modified so that the list of modes comes from cfg/dos.py.
- Merge pull request #158 from alex2shiyu/master.

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