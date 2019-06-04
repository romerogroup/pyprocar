PyProcar
===========

PyProcar is a robust, open-source Python library used for pre- and post-processing of the electronic structure data coming from DFT calculations. PyProcar provides a set of functions that manage the PROCAR file obtained from Vasp and Abinit. Basically, the PROCAR file is a projection of the Kohn-Sham states over atomic orbitals. That projection is performed to every k-point in the considered mesh, every energy band and every atom. PyProcar is capable of performing a multitude of tasks including plotting plain and spin/atom/orbital projected band structures and Fermi surfaces- both in 2D and 3D, Fermi velocity plots, unfolding bands of a super  cell, comparing band structures from multiple DFT calculations and generating a k-path for a given crystal structure. Currently supports VASP and Abinit. 


Usage
-----
Typical use is as follows

    import pyprocar 
    pyprocar.bandsplot('PROCAR',outcar='OUTCAR',mode='plain')

Refer to the documentation for further details. 

Documentation
-------------

https://romerogroup.github.io/pyprocar/

Contributors
------------
Francisco Munoz <br />
Aldo Romero <br />
Sobhit Singh <br />
Uthpala Herath <br />
Pedram Tavadze <br />
Eric Bousquet <br />
Xu He <br />
Jordan Bieder <br />

Mailing list
-------------
Please post your questions on our forum.

https://groups.google.com/d/forum/pyprocar

Dependencies
------------
matplotlib <br />
numpy <br />
scipy <br />
seekpath <br />
ase <br />
scikit-image <br />

Installation
------------

	pip install pyprocar	

Changelog
--------------
v0.1.0, June 10, 2013 -- Initial release.

v2.0 Mar 21,2018 -- Created PyProcar package version with added support to Abinit. 

v2.1 Apr 03,2018 -- Fixed issue with input arguments when using OUTCAR as an input 

v2.2 May 14,2018 -- Updated documentation.

v2.3 May 17,2018 -- Added k path generator.

v2.4 May 18,2018 -- Fixed minor issues with fermi2D and procarsymmetry

v2.5 May 18.2018 -- Fixed issue with Vector

v2.6 May 18,2018 -- Fixed more issues with fermi2D

v2.7 May 18,2018 -- Fixed out-of-bounds error in k path generator.

v2.8 May 23,2018 -- Fixed procar.cat()

v2.9 Jul 29,2018 -- Created PyProcar Mailing list.

v3.0 Sep 17, 2018 -- Added method to compare two PROCARs. Moved to Python3. 

v3.1 Sep 19, 2018 -- Minor bug fixes. 

v3.2 Nov 26, 2018 -- Moved project to romerogroup.

v3.3 Mar 19, 2019 -- Added band unfolder. 

v3.4 May 21, 2019 -- Bug fixes for plotting and added capability to plot meta-GGA. 

v3.5 May 22, 2019 -- added automatic high symmetry point labeling from KPOINTS file.

v3.6 Jun 04, 2019 -- Added 3D Fermi surface utility.

v3.7 Jun 04, 2019 -- Bug fixes for Fermi surface.

v3.71 Jun 05, 2019 -- More bug fixes. 

v3.8 Jun 05, 2019 -- Updated reading from gzip for binary data. Increased parsing speed when phase factors are present. 

