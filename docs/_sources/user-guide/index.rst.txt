User Guide
**********
In this section, we will provide an overview of the information obtained from DFT codes and the methods used by Pyprocar to access this data. 
For guidance on conducting DFT calculations to generate the necessary files for running PyProcar, please refer to the :ref:`dftprep` section. 
PyProcar is able to process data from various codes, and the format of the information remains consistent across them. 
As an illustration, we have included an example of the atomic projections commonly found in DFT codes, using data from vasp.


The format of the PROCAR is as follows::

	1.   PROCAR lm decomposed 
	2.    of k-points:    4         # of bands: 224         # of ions:  4 
	3. 
	4.    k-point    1 :    0.12500000 0.12500000 0.12500000     weight = 0.12500000 
	5.   
	6.   band   1 # energy  -52.65660295 # occ.  1.00000000 
	7.   
	8.   ion      s     py     pz     px    dxy    dyz    dz2    dxz    dx2    tot 
	9.     1  0.052  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.052
	10.    2  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000 
	11.    3  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000 
	12.    4  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000 
	13.    4  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000 
	14.   tot 0.052  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.052 


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


The site projected wave function in the PROCAR is calculated by projecting  the  Kohn-Sham  wave  functions  onto  spherical  harmonics  that  are non-zero  within  spheres  of  a  Wigner-Seitz  radius  around  each  ion  by:

.. math::

   |<Y^{\alpha}_{lm}|\phi_{nk}>|^2

where, 
:math:`Y^{\alpha}_{lm}` are the  spherical harmonics centered at ion index :math:`\alpha` with angular moment :math:`l` and magnetic quantum number :math:`m`, and :math:`\phi_{nk}` are the Kohn-Sham wave functions.  In general, for a non-collinear electronic structure calculation the same equation is generalized to:

.. math::

	\frac{1}{2} \sum_{\mu, \nu=1}^{2} \sigma_{\mu, \nu}^{i}<\psi_{n, k}^{\mu}\left|Y_{l m}^{\alpha}><Y_{l m}^{\alpha}\right| \psi_{n, k}^{\nu}>

where :math:`\sigma^i` are the Pauli matrices with :math:`i = x, y , z` and the spinor wavefunction :math:`\phi_{nk}` is now defined as 	 

.. math::
	
	\phi_{nk} & = \begin{bmatrix}
	\psi_{nk}^{\uparrow} \\
	\psi_{nk}^{\downarrow}
	\end{bmatrix}


Further Details
===============
.. toctree::
   :maxdepth: 2

   atomic_projections
   bands
   cat
   comparebands
   dos
   ebs
   fermi2d
   fermi3d
   filter
   kpath
   repair
   structure
   unfold