Band structure
===================

PyProcar goes beyond the conventional plain band structure to plot the projected bands that carry even more information. The projected bands are color coded in an informative manner to portray fine details. 

=======================
1. Plain band structure
=======================

This is the most basic type of band structure. No projection information is contained here. In order to use the plain mode one sets ``mode=`plain'``. ``elimit`` sets the energy window limits. ``outcar`` specifies the **OUTCAR** file. For Abinit calculations, ``abinit_output`` is used instead. ``color`` lets the user use any color available in the matplotlib package. If an output file is not present one can set ``fermi`` manually. One may save the plot using the ``savefig`` tag, for example, ``savefig='figure.png'`` with a desired image file format. This applies to all other band structure plotting functions in PyProcar as well. 

Usage::

	pyprocar.bandsplot('PROCAR',outcar='OUTCAR',elimit=[-2,2],mode='plain',color='blue') 

PyProcar is capable of labeling the :math:`k`-path names automatically, however, the user can manually input them as desired. 

If a **KPOINTS** file is present automatic labeling can be enabled as follows::

	pyprocar.bandsplot('PROCAR',outcar='OUTCAR',elimit=[-2,2],mode='plain',color='blue',kpointsfile='KPOINTS')

One may manually label the :math:`k`-path as well. ``knames`` and ``kticks`` corresponds to the labels and the number of grid points between the high symmetry points in the :math:`k`-path used for the band structure calculation. Usage::

	pyprocar.bandsplot('PROCAR',outcar='OUTCAR',elimit=[-2,2],mode='plain',color='blue',kticks=[0,39,79,119,159],knames=['G','X','M','G','R'])

This is valid for the rest of the band plotting projections and also for the ``bandscompare()`` function.

==================
2. Spin projection
==================

For collinear spin polarized and non-collinear spin calculations of DFT codes, PyProcar is able to plot the bands of each spin channel or direction separately. For the former case setting ``spin=0`` plots the unpolarized bands  and ``spin=1`` plots the spin channels separately. 

For non-collinear spin calculations, spin=1,2,3 corresponds to spins oriented in :math:`S_x`, :math:`S_y` and :math:`S_z` directions respectively. Setting ``spin='st'`` plots the spin texture perpendicular in the plane (:math:`k_x`,:math:`k_y`) to each (:math:`k_x`,:math:`k_y`) vector. This is useful for Rashba-like states in surfaces. For parametric plots such as spin, atom and orbitals, the user should set ``mode=`parametric'``. ``cmap`` refers to the matplotlib color map used for the parametric plotting and can be modified by using the same color maps used in matplotlib.

Usage::

	pyprocar.bandsplot('PROCAR',outcar='OUTCAR',elimit=[-5,5],kticks=[0,39,79,119,159],knames=['G','X','M','G','R'],cmap='jet',mode='parametric',spin=1)

If spin-up and spin-down bands are to be plot separately, one may use the ``filter()`` function to create two PROCARs for each case and plot them individually. An example is given below::

	pyprocar.filter('PROCAR','PROCAR-up',spin=[0])
	pyprocar.filter('PROCAR','PROCAR-down',spin=[1])
	pyprocar.bandsplot('PROCAR-up',...)
	pyprocar.bandsplot('PROCAR-down',...)


==================
3. Atom projection
==================

The projection of atoms onto bands can provide information such as which atoms contribute to the electronic states near the Fermi level. PyProcar counts each row of ions in the PROCAR file, starting from zero. In an example of a five atom SrVO_3, the indexes of ``atoms`` for Sr, V and the three O atoms would be 1,2 and 3,4,5 respectively. It is also possible to include more than one type of atom by using an array such as ``atoms = [1,2,3]``.

Usage::

	pyprocar.bandsplot('PROCAR',outcar='OUTCAR',elimit=[-5,5],kticks=[0,39,79,119,159],knames=['G','X','M','G','R'],cmap='jet', mode='parametric',atoms=[1])

=====================
4. Orbital projection
=====================

The projection of atomic orbitals onto bands is also useful to identify the contribution of orbitals to bands. For instance, to identify correlated :math:`d` or :math:`f` orbitals in a strongly correlated material near the Fermi level. It is possible to include more than one type of orbital projection. The mapping of the index of orbitals to be used in ``orbitals`` is as follows (this is the same order from the PROCAR file). 

.. image:: orbitals.png

Usage: To project all five :math:`d`-orbitals:: 

	pyprocar.bandsplot('PROCAR',outcar='OUTCAR',elimit=[-5,5],kticks=[0,39,79,119,159],knames=['G','X','M','G','R'],cmap='jet',mode='parametric',orbitals=[4,5,6,7,8])

One or many of the above can be combined together to allow the user to probe into more specific queries such as a collinear spin projection of a certain orbital of a certain atom. 

Different modes of band structures are useful for obtaining information for different cases. The four modes available within PyProcar are ``plain, scatter, parametric`` and ``atomic``. The ``plain`` bands contain no projection information. The ``scatter`` mode creates a scatter plot of points. The ``parametric`` mode interpolates between points to create bands which are also projectable. Finally, the ``atomic`` mode is useful to plot energy levels for atoms. To  set  maximum  and  minimum  projections  for  color  map,  one  could  use ``vmin`` and ``vmax`` tags.


.. automodule:: pyprocar.scriptBandsplot
	:members:

