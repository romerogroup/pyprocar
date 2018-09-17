PyProcar
===========

PyProcar provides a set of scripts that manage the PROCAR file obtained from Vasp and Abinit. Basically, the PROCAR
file is a projection of the Kohn-Sham states over atomic orbitals. That projection is performed to every K
point in the considered mesh, every energy band and every atom. Here you will find scripts that help
you in digging the information from it and plot it in a nice and friendly process.


Usage
-----
Typical use is as follows

    #!/usr/bin/env python
    import pyprocar 
    pyprocar.bandsplot('PROCAR',outcar='OUTCAR',mode='parametric')

Refer to the documentation for further details. 

Contributors
------------
Francisco Munoz <br />
Aldo Romero <br />
Sobhit Singh <br />
Uthpala Herath <br />
Pedram Tavadze <br />
Eric Bousquet <br />
Xu He <br />

Mailing list
-------------
Please post your questions on our forum.

https://groups.google.com/d/forum/pyprocar

Changes
-------
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

v3.0 Sep 17, 2018 -- Added method to compare two PROCARs.
		     Moved to Python3. 


Installation
------------

	pip3 install pyprocar33	
