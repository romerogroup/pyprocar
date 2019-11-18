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

How to cite
-----------
If you have used PyProcar in your work, please cite: 

[arXiv:1906.11387 [cond-mat.mtrl-sci]](https://arxiv.org/abs/1906.11387)

Thank you.

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

v3.8.1 Jun 05, 2019 -- Updated reading from gzip for binary data. Increased parsing speed when phase factors are present. 

v3.8.2 Jun 05, 2019 -- Updated docs. 

v3.8.3 Jun 05, 2019 -- Updated parsing for PROCAR with phase. 

v3.8.4 Jun 11, 2019 -- Fixed parsing old PROCAR format. 

v3.8.5 Jun 13, 2019 -- Bug fixes in Fermi surface plotting. 

v3.8.6 Jun 26, 2019 -- Bug fixes in band unfolding Fermi shift energy and band structure labels for Fermi shifts.  

v3.8.7 Jul 21, 2019 -- Fixed bug in K-mesh generator. 

v3.8.8 Jul 24, 2019 -- Fixed ambiguity in spin flag. 

v3.8.9 Sep 9, 2019 -- Added bbox_inches='tight' for savefig.

v3.9.0 Sep 12, 2019 -- Fixed spin polarized band unfolding.  

v3.9.1 Sep 15, 2019 -- Fixed unfold spin polarized eigenvalue bug and spin up/down band energy error in unfolding. 

v3.9.2 Oct 4, 2019 -- Fixed bug in 2D Kmesh generator. 

v4.0.0 Nov 6th, 2019 -- Various bug fixes. Release of standalone version. Updated documentation.

v4.0.1 Nov 17th, 2019 -- Added feature to filter k-points.  