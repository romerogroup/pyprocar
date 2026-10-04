.. _ebs:

ElectronicBandStructure
=======================

PyProcar provides a centralized data class to manage the electronic band structure information from various DFT codes, 
known as the ElectronicBandStructure class. This class takes kpoints, bands, and efermi as arguments, 
which are the essential requirements for plotting a band-like object. 
Additionally, it can accept other keyword arguments, such as projected, projected_phase, Kpath, weights, 
labels, reciprocal_lattice, and shifted_to_efermi.

Accessing Electronic Band Structure
+++++++++++++++++++++++++++++++++++++

The ElectronicBandStructure object (referred to as "ebs") can be accessed through the parser that io.get_parser returns:

.. code-block:: python

    import pyprocar

    parser = pyprocar.io.get_parser('vasp', path_to_calculation)
    ebs = parser.ebs

Using the ebs object, you can access various information related to the electronic band structure:

.. code-block:: python

    ebs.kpoints # kpoints in the reduced basis
    ebs.bands # bands, shape (n_kpoints, n_bands, n_spin_channels)
    ebs.fermi # The fermi energy

    ebs.projected # The atomic projections, shape (n_kpoints, n_bands, n_spins, n_atoms, n_orbitals)
    ebs.projected_phase # The complex atomic projections
    ebs.weights # The kpoint weights
    ebs.orbital_names # The name of each orbital in the projections

    ebs.n_kpoints # The number of k points
    ebs.n_bands # The number of bands
    ebs.n_atoms # The number of atoms
    ebs.n_orbitals # The number of orbitals
    ebs.n_spins # The number of spin projections
    ebs.is_non_collinear # Boolean if this is a non-collinear calculation

    ebs.kpoints_cartesian # The kpoints in cartesian coordinates

    # A band structure along a path (ElectronicBandStructurePath) also has
    ebs.kpath # The kpath information
    ebs.tick_names # The labels of the high-symmetry points

    # A band structure on a grid (ElectronicBandStructureMesh) also has
    ebs.n_kx # Unique kpoints along the k1 direction
    ebs.n_ky # Unique kpoints along the k2 direction
    ebs.n_kz # Unique kpoints along the k3 direction
    ebs.get_kpoints_mesh() # The kpoints as a (n_kx, n_ky, n_kz, 3) grid
    ebs.get_property_mesh("bands") # A property, such as the bands, on that grid

    # Useful methods
    ebs.ebs_sum(atoms=None, orbitals=None, spins=None) # Sum the projections over atoms, orbitals and spins

An ``ElectronicBandStructureMesh`` built from a calculation that used symmetry holds only the
irreducible k-points the code computed, so it rebuilds the full grid when it is created. It
carries each value to its images with the symmetry operations of the structure, permuting atoms,
rotating orbitals and the non-collinear spin, and conjugating the phases under time reversal. To
do this by hand on an irreducible band structure, give the grid the code reduced. The symmetry
operations default to those of ``ebs.structure``; ``rotations`` and ``time_reversals`` override
them.

.. code-block:: python

    from pyprocar.core.ebs import ibz2fbz
    from pyprocar.core.kpoints import KGRID_MODE, KGridInfo

    grid = KGridInfo(kgrid=(15, 15, 15), kgrid_mode=KGRID_MODE.GAMMA, kshift=(0, 0, 0))
    ibz2fbz(ebs, kgrid_info=grid)
