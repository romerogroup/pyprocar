.. _atomic_projections:

Atomic Projections
==================

PyProcar specializes in plotting the atomic projections, 
thus there are some conventions we keep consistent through when accessing these projections.
These can be applied to the bands structure, density of states, and fermi surfaces.

==================
1. Spin projection
==================

For collinear spin polarized and non-collinear spin calculations of DFT codes, PyProcar is able to plot the bands considering spin density (magnitude), spin magnetization and spin channels separately.

For non-collinear spin calculations, ``spins=[0]`` plots the spin density (magnitude) and ``spins=[1,2,3]`` corresponds to spins oriented in :math:`S_x`, :math:`S_y` and :math:`S_z` directions respectively. 
Setting ``spin='st'`` plots the spin texture perpendicular in the plane (:math:`k_x`, :math:`k_y`) to each (:math:`k_x`,i :math:`k_y`) vector. 
This is useful for Rashba-like states in surfaces. For parametric plots such as spin, atom and orbitals, the user should set ``mode=`parametric'``. ``cmap`` refers 
to the matplotlib color map used for the parametric plotting and can be modified by using the same color maps used in matplotlib. ``cmap='seismic'`` is recommended for parametric spin band structure plots.  
For colinear spin calculations setting ``spins=[0]`` plots the spin density (magnitude) and ``spins=[1]`` plots the spin magnetization. Spin channels can also be plot separately (see below).

Currently, Elk only supports spin colinear plotting.

==================
2. Atom projection
==================

The projection of atoms onto bands can provide information such as which atoms contribute to the electronic states near the Fermi level. 
PyProcar counts each row of ions in the PROCAR file, starting from zero. In an example of a five atom SrVO:math:`_3`, the indexes of ``atoms`` for Sr, V and the three O atoms would be 0,1 and 2,3,4 respectively.
It is also possible to include more than one type of atom by using an array such as ``atoms = [0,1,3]``. 


=====================
3. Orbital projection
=====================

The projection of atomic orbitals onto bands is also useful to identify the contribution of orbitals to bands. 
For instance, to identify correlated :math:`d` or :math:`f` orbitals in a strongly correlated material near the Fermi level. 
It is possible to include more than one type of orbital projection. 
The mapping of the index of orbitals to be used in ``orbitals`` is as follows (this is the same order from the PROCAR file). 
Quantum Espresso, VASP and Abinit follows this order. 

.. image:: ../images/orbitals.png

In Quantum Espresso when there is a noncolinear spin-orbit calculation, the orbitals will be in the JM basis, therefore the index mapping of the orbitals will be the following.

.. code-block::
    :caption: JM orbital mapping

    orbitals = [
                    {"l": 's', "j": 0.5, "m": -0.5}, -> 0
                    {"l": 's', "j": 0.5, "m": 0.5},  -> 1

                    {"l": 'p', "j": 0.5, "m": -0.5}, -> 2
                    {"l": 'p', "j": 0.5, "m": 0.5},  -> 3

                    {"l": 'p', "j": 1.5, "m": -1.5}, -> 4
                    {"l": 'p', "j": 1.5, "m": -0.5}, -> 5
                    {"l": 'p', "j": 1.5, "m": -0.5}, -> 6
                    {"l": 'p', "j": 1.5, "m": 1.5},  -> 7

                    {"l": 'd', "j": 1.5, "m": -1.5}, -> 8
                    {"l": 'd', "j": 1.5, "m": -0.5}, -> 9
                    {"l": 'd', "j": 1.5, "m": -0.5}, -> 10
                    {"l": 'd', "j": 1.5, "m": 1.5},  -> 11

                    {"l": 'd', "j": 2.5, "m": -2.5}, -> 12
                    {"l": 'd', "j": 2.5, "m": -1.5}, -> 13
                    {"l": 'd', "j": 2.5, "m": -0.5}, -> 14
                    {"l": 'd', "j": 2.5, "m": 0.5},  -> 15
                    {"l": 'd', "j": 2.5, "m": 1.5},  -> 16
                    {"l": 'd', "j": 2.5, "m": 2.5},  -> 17
                ]

In Elk, the :math:`Y_{lm}` projections of the atomic site resolved DOS are arranged in logical order in the BAND_S*A* files, namely: (l,m) = (0,0), (1,-1), (1,0), (1,1), (2,-2), (2,-1), (2,0), (2,1), (2,2), etc., 

One or many of the above can be combined together to allow the user to probe into more specific queries such as a collinear spin projection of a certain orbital of a certain atom.
