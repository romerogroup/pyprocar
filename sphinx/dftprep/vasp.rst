.. _vasp: 

VASP Preparation
==============================================

This guide is here to help you prepare VASP Calculations to be used with pyprocar.

- Required files : PROCAR, OUTCAR (optional), KPOINTS (optional)
- flag           : code='vasp' (default)

In the VASP code, the wavefunction projection information is written into the PROCAR file when ``LORBIT=11`` is set in the INCAR file. For band unfolding, set ``LORBIT=12`` to include phase projections of the wave functions.
An OUTCAR file is required to extract the Fermi-energy and reciprocal lattice vectors. If a KPOINTS file is provided, the :math:`k`-path will automatically be labeled in the band structure.
To perform spin colinear calculations set ``ISPIN = 2`` in the ``INCAR``.
To perform spin non-colinear calculations set ``ISPIN = 2`` and ``LNONCOLLINEAR = .TRUE.``.

First perform a self-consistent calculation with a :math:`k`-mesh grid. Then set ``ICHARG=11`` in the INCAR and create a KPOINTS file containing the :math:-`k`-path. This can be done with the :ref:`kpath` feature in PyProcar. 
