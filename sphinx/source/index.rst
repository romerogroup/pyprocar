PyProcar
======================

.. toctree::
	:hidden:

	getting-started/index
	user-guide/index
	dftprep/index
	examples/index
	api/index
	

.. raw:: html

    <div class="banner">
		<h3>Pre- and Post- processing of Density Functional Theory Codes</h2>
	 	<a href="./examples/index.html"><center><img src="_static/images/welcome.png" alt="pyprocar" width="75%"/></a>
    </div>


PyProcar is a robust, open-source Python library used for pre- and post-processing of the electronic structure data coming from DFT calculations. 
PyProcar provides a set of functions that manage data from the PROCAR format obtained from various DFT codes. 
Basically, the PROCAR file is a projection of the Kohn-Sham states over atomic orbitals. 
That projection is performed to every :math:`k`-point in the considered mesh, every energy band and every atom. 
PyProcar is capable of performing a multitude of tasks including plotting plain and spin/atom/orbital projected band structures and 
Fermi surfaces- both in 2D and 3D, Fermi velocity plots, unfolding bands of a super  cell, comparing band structures from multiple DFT calculations, 
plotting partial density of states and generating a :math:`k`-path for a given crystal structure. 

Currently supports:

1. VASP
2. Quantum Espresso
3. Abinit
4. Elk (Band Structure and Fermi in development)
5. Lobster (Stll in development)


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


