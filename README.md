PyPROCAR
===========

Prprocar provides a set of scripts that manage the PROCAR file obtained from Vasp and Abinit. Basically, the PROCAR
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
Francisco Munoz
Aldo Romero
Sobhit Singh
Uthpala Herath
Pedram Tavadze
Eric Bousquet 
Xu He

Changes
-------
v0.1.0, June 10, 2013 -- Initial release.

v2.0 Mar21,2018 -- Created PyProcar package version with added support to Abinit. 


Installation
------------

From SOURCEFORGE (We will be shifting to github soon.)
	-Download the pyprocar package and cd into the root directory.Then,

	>sudo python setup.py install


From PyPI

	>pip install pyprocar	
