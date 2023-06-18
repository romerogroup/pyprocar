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

The ElectronicBandStructure object (referred to as "ebs") can be accessed through the main io.Parser class:

.. code-block:: python

    import pyprocar

    parser = pyprocar.io.Parser(code = 'vasp', dir=path_to_calculation)
    ebs = parser.ebs

Using the ebs object, you can access various information related to the electronic band structure:

.. code-block:: python

    ebs.kpoints # kpoints in the reduced basis
    ebs.bands # bands in the reduced basis
    ebs.efermi # The fermi energy

    ebs.projected # The atomic projections array
    ebs.projected_phase # The complex atomic projections array
    ebs.kpath # The kpath information
    ebs.labels # The kpath labels
    ebs.weights # The kpoint weights

    ebs.n_kx # Unique kpoints along the k1 direction
    ebs.n_ky # Unique kpoints along the k2 direction
    ebs.n_kz # Unique kpoints along the k3 direction
    ebs.nkpoints # The number of k points
    ebs.nbands # The number of bands
    ebs.natoms # The number of atoms
    ebs.nprincipal # The number of the prinicipal quantum number
    ebs.norbitals # The number of orbitals 
    ebs.nspins # The number of spins

    ebs.is_non_collinear # Boolean if this is a non-collinear calcuulation

    ebs.kpoints_cartesian # The kpoints in cartesian coordinates
    ebs.kpoints_reduced # The kpoints in reduced coordinates


    # Sometimes having kpoint infomation in mesh grid can be useful. So the following attributes are in the form of a meshgrid
    ebs.index_mesh # The index mesh store the kpoint index in the original kpoints list at particular grid point
    ebs.kpoints_mesh # Kpoint mesh representation of the kpoints grid.
    ebs.cartesian_mesh # Kpoint cartesian mesh representation of the kpoints grid.
    ebs.bands_mesh # Bands mesh is a numpy array that stores each band in a mesh grid.
    ebs.projected_mesh # projected mesh is a numpy array that stores each projection in a mesh grid.
    ebs.project_phase_mesh # projected phase mesh is a numpy array that stores each projection phases in a mesh grid.
    ebs.weights_mesh # weights mesh is a numpy array that stores each weights in a mesh grid. 

    ebs.bands_gradient_mesh # Bands gradient mesh is a numpy array that stores each band gradient in a mesh grid.
    ebs.bands_hessian_mesh # Bands hessian mesh is a numpy array that stores each band gradient in a mesh grid.

    # Useful methods
    ebs.ebs_sum(atoms,orbitals,spins) # Sum the atomic projections over the atoms, orbitals, spins, and prinicipal
    ebs.ibz2fbz(rotations=rotations) # if the calculation used symmetry this method will recover the full information of the broullin zone based on the symmetry rotations