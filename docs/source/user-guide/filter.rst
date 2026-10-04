.. _filter:


Filtering data
==============

A simpler version of PROCAR file containing only a subset of information from the original dataset can be generated with this utility.  This feature is very useful when there are many bands in the PROCAR file (e.g. in heterostructures or supercells calculations) making the file size enormously large for post-processing while only bands near the Fermi-level are needed for analysis. In this case, one can filter data of selected bands near the Fermi-level. This considerably reduces the file size and makes the post-processing of data faster. In the same way, one could use the ``filter`` utility to filter the **PROCAR** file to extract information regarding particular spins, atoms, or orbitals in a relatively smaller **PROCAR-new** file.

The following example extracts information of bands ranging from index 50 to 70 from a **PROCAR** file (Fermi-level is near band \#60) while ignoring all other bands located far from the Fermi-level, and stores resulting dataset in a new file named **PROCAR-band50-70**. Now the new **PROCAR-band50-70** file can be used for further post-processing of data at relatively low memory requirements::

	pyprocar.filter('PROCAR','PROCAR-band50-70',bands=[50,70])

===========================
To filter selected orbitals
===========================

To make a new PROCAR file containing only three columns, one for :math:`s` orbitals, one for :math:`p`, and one for total (real total, not total of :math:`s+p`)::

	pyprocar.filter('PROCAR’,'PROCAR-filtered_sp’, orbitals=[[0],[1,2,3]])

If you want to select only :math:`p_x` orbitals, just use ``orbitals=[[3]]``.

In the same way to plot the projection of total :math:`p`-orbitals use ``orbitals=[[1,2,3]]``  (i.e. :math:`px+py+pz`). Same goes for other orbitals.

Each new column is named by ``orbital_names``, which defaults to ``o0``, ``o1``, and so on. PyProcar reads these names from the header of the new file, so plots label the columns ``o0`` and ``o1``. In the new file, orbital index ``i`` selects the ``i``-th column, so ``orbitals=[1]`` selects the :math:`p` sum of the example above. Pass ``orbital_names=['s','p']`` to label the columns ``s`` and ``p``.

A shell name such as ``orbitals=['p']`` selects the columns whose names make up the whole :math:`p` shell, for example ``py``, ``pz`` and ``px``. Columns named ``o0``, ``o1`` or ``p`` make up no shell, so select them by index.

===================================
To filter selected :math:`k`-points
===================================

This is helpful especially for hybrid calculations where the weighted :math:`k`-points should be filtered to perform band structure plotting.
To filter :math:`k`-points within a minimum and maximum range::

	pyprocar.filter('PROCAR’,'PROCAR-filtered_kpoints’, kpoints=[10,13])

This creates a new PROCAR file containing only the selected :math:`k`-points.

========================
To filter selected spins
========================

This can be used to generate a PROCAR that contains a selected spin component.
For non colinear spin calculations 0 is the total spin density and 1,2,3 correspond to spins in the x,y and z directions.
For example, to filter the spin in the :math:`S_x` direction::

	pyprocar.filter('PROCAR’,'PROCAR-filtered_spin’, spin=[1])

This creates a new PROCAR file containing only the selected spin component(s).
For colinear spin calculations spin=[0] and spin=[1] selects spin up and spin down components, respectively.


========================
To filter selected atoms
========================

Usage::

	pyprocar.filter(‘PROCAR’,’PROCAR-filter_ATOMS’, atoms=[[0]])

PyProcar counts each row of ions in PROCAR file starting from 0. Keep in mind: ``atoms=0`` does not define the atom_type, rather it defines only the first ion in the POSCAR. So if you have more than one ion of the same element, use ``atoms = [[0,1,2,3,...]]``.

Each group in ``atoms`` becomes one row of the new PROCAR, the sum of the rows of its atoms. For a SrVO3 calculation with the POSCAR order Sr, V, O, O, O::

	pyprocar.filter('PROCAR','PROCAR-filtered_atoms', atoms=[[0],[1],[2,3,4]])

writes 3 rows: Sr, V, and the sum of the three O atoms. To plot the new file, rename it to ``PROCAR`` in the calculation directory. When the PROCAR has fewer or more rows than the POSCAR has atoms, PyProcar warns that the PROCAR was filtered by atoms and treats each row as a group:

- Atom index ``i`` selects the ``i``-th group, so ``atoms=[2]`` selects the three O atoms. Plots label a selection by its group indices, not by a species.
- A plot with no ``atoms`` sums every group.
- A selection by species raises an error, because a group can hold atoms of several species. This includes ``atoms=['O']`` and the ``overlay_species`` mode. Select groups by index instead.
- The lattice and the symmetry operations still come from the POSCAR and the OUTCAR, so Fermi surfaces unfold as before.

PyProcar cannot detect a filtered PROCAR that has exactly as many rows as the POSCAR has atoms. It then labels each row with the species of the POSCAR atom at the same index.

.. automodule:: pyprocar.scripts.scriptFilter
	:members:
