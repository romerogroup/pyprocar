Generating a k-path
===================


In order to plot a band structure, one must define a set of :math:`k`-points following a desired :math:`k`-path in momentum space. PyProcar’s :math:`k`-path generation utility enables a the user to automatically generate a suitable and sufficient :math:`k`-path given the crystal structure, typically read from the POSCAR file which is displayed below.

::

	Sb Bi                                   
	   4.51004000000000     
	   0.8660254037844390   -0.5000000000000000    0.0000000000000000
	   0.0000000000000000    1.0000000000000000    0.0000000000000000
	   0.0000000000000000    0.0000000000000000    2.6420852143218241
	   3     3
	Direct
	  0.0000000000000000  0.0000000000000000  0.6470799999999988
	  0.6666666666666643  0.3333333333333357  0.9804133333333345
	  0.3333333333333357  0.6666666666666643  0.3137466666666702
	  0.0000000000000000  0.0000000000000000  0.1818699999999997
	  0.6666666666666643  0.3333333333333357  0.5152033333333354
	  0.3333333333333357  0.6666666666666643  0.8485366666666640

General format::

	pyprocar.kpath(infile, grid-size, with-time-reversal, recipe, threshold, symprec, angle-tolerance,supercell_matrix)

Usage::
	
	pyprocar.kpath(`POSCAR',40,True,`hpkot',1e-07,1e-05,-1.0,np.eye(3))	

This information is automatically written to a KPOINTS file.

More details regarding these parameters can be found in the `SeeK-path manual <https://seekpath.readthedocs.io/en/latest/module_guide/index.html>`_.
The :math:`k`-path generation utility within PyProcar is based on the Python library **seekpath** developed by Hinuma et al::

	Y. Hinuma, G. Pizzi, Y. Kumagai, F. Oba, I. Tanaka, Band structure diagram paths based on crystallography, Computational Materials Science 128 (2017) 140–184.doi:10.1016/j.commatsci.2016.10.015.


.. automodule:: pyprocar.scriptKpath
	:members:



