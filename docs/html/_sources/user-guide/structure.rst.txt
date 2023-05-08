.. _structure:


Structure
===============

PyProcar provides a data class to manage the structure information of the DFT calculation, 
known as the Structure class. This class takes atoms, cartesian_coordinates, 
fractional_coordinates, lattice, rotations as arguments.


Accessing Structure Information
+++++++++++++++++++++++++++++++++++++

The Structure object (referred to as "structure") can be accessed through the main io.Parser class:

.. code-block:: python

    import pyprocar

    parser = pyprocar.io.Parser(code = 'vasp', dir=path_to_calculation)

    structure = parser.structure

Using the structure object, you can access various information related to the structure:

.. code-block:: python

    structure.volume
    structure.density
    structure.a
    structure.b
    structure.c
    structure.alpha
    structure.beta
    structure.gamma
    structure.nspecies
    structure.natoms
    structure.atomic_numbers
    structure.reciprocal_lattice
    structure.lattice_corners
    structure.cell_convex_hull


    structure.get_space_group_number()
    structure.get_space_group_international()
    structure.get_wyckoff_positions()
    structure.get_spglib_symmetry_dataset()
    

