PyProcar documentation
======================
PyProcar is a robust, open-source Python library used for pre- and post-processing of the electronic structure data coming from DFT calculations. PyProcar provides a set of functions that manage the PROCAR file obtained from Vasp and Abinit. Basically, the PROCAR file is a projection of the Kohn-Sham states over atomic orbitals. That projection is performed to every :math:`k`-point in the considered mesh, every energy band and every atom. PyProcar is capable of performing a multitude of tasks including plotting plain and spin/atom/orbital projected band structures and Fermi surfaces- both in 2D and 3D, Fermi velocity plots, unfolding bands of a super  cell, comparing band structures from multiple DFT calculations and generating a :math:`k`-path for a given crystal structure. In the VASP code, this information is written into the PROCAR file when ``LORBIT=11`` or ``LORBIT=12`` (to include phase projections of the wave functions) in the INCAR file.

The format of the PROCAR is as follows::

	1.   PROCAR lm decomposed 
	2.    of k-points:    4         # of bands: 224         # of ions:  4 
	3. 
	4.    k-point    1 :    0.12500000 0.12500000 0.12500000     weight = 0.12500000 
	5.   
	6.   band   1 # energy  -52.65660295 # occ.  1.00000000 
	7.   
	8.   ion      s     py     pz     px    dxy    dyz    dz2    dxz    dx2    tot 
	9.     1  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000 
	10.    2  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000 
	11.    3  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000 
	12.    4  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000 
	13.    4  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000 
	14.   tot 0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000 


- Line 1 is a comment 
- Line 2 gives the number of K points calculated (kpoint), number of bands (nband) and number of ions (nion) 
- Line 4 gives the k-point and the weight
- Line 6 gives the energy for kpoints 
- Line 8 Labels of calculated projections, column 11 is the total projection
- Line 9 Calculated projections for atom 1 
- Line 10 Calculated projections for atom 2 and so on 
- Line 14 after projections over all atoms, the total over every atomic projection is reported 

This block is repeated for the other spin channel if the calculation was polarized. 
For spin polarized or non-collinear spin calculations there are additional blocks for each spin component.


The site projected wave function in the PROCAR file is calculated by projecting  the  Kohn-Sham  wave  functions  onto  spherical  harmonics  that  are non-zero  within  spheres  of  a  Wigner-Seitz  radius  around  each  ion  by:

.. math::

   |<Y^{\alpha}_{lm}|\phi_{nk}>|^2

where, 
:math:`Y^{\alpha}_{lm}` are the  spherical harmonics centered at ion index :math:`\alpha` with angular moment :math:`l` and magnetic quantum number :math:`m`, and :math:`\phi_{nk}` are the Kohn-Sham wave functions.  In general, for a non-collinear electronic structure calculation the same equation is generalized to:

.. math::

	\frac{1}{2} \sum_{\alpha,\beta = 1}^2 \sigma_{\alpha,\beta}^i
	<{\psi_{n,k}^\alpha | Y_{lm}^{\alpha}}> 
	<{Y_{lm}^{\beta} | \psi_{n,k}^\beta}>

where :math:`\sigma^i` are the Pauli matrices with :math:`i = x, y , z` and the spinor wavefunction :math:`\phi_{nk}` is now defined as 	 

.. math::
	
	\phi_{nk} & = \begin{bmatrix}
	\psi_{nk}^{\uparrow} \\
	\psi_{nk}^{\downarrow}
	\end{bmatrix}

An OUTCAR file or an equivalent Abinit output file is required to extract the Fermi-energy and reciprocal lattice vectors. 


.. toctree::
   :maxdepth: 2
      
   installation
   contributors
   cite
   tutorials

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
