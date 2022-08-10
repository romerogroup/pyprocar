.. _labelkpath:

Generating a k-path
===================


In order to plot a band structure, one must define a set of :math:`k`-points following a desired :math:`k`-path in momentum space. PyProcar’s :math:`k`-path generation utility enables a the user to automatically generate a suitable and sufficient :math:`k`-path given the crystal structure, typically read from the POSCAR file (VASP). 

General format::

	pyprocar.kpath(infile, outfile, grid-size, with-time-reversal, recipe, threshold, symprec, angle-tolerance,supercell_matrix)

Usage::
	
	pyprocar.kpath('POSCAR','KPOINTS',40,True,'hpkot',1e-07,1e-05,-1.0,np.eye(3))	

or using the default options, a POSCAR would suffice::
    
    pyprocar.kpath('POSCAR')

This information is automatically written to a KPOINTS file. The retrieved :math:`k`-path can be used for other DFT codes with slight modifications.

More details regarding these parameters can be found in the `SeeK-path manual <https://seekpath.readthedocs.io/en/latest/module_guide/index.html>`_.
The :math:`k`-path generation utility within PyProcar is based on the Python library **seekpath** developed by Hinuma et al::

	Y. Hinuma, G. Pizzi, Y. Kumagai, F. Oba, I. Tanaka, Band structure diagram paths based on crystallography, Computational Materials Science 128 (2017) 140–184.doi:10.1016/j.commatsci.2016.10.015.


.. automodule:: pyprocar.scriptKpath
	:members:



