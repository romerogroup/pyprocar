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


Preparing Calculations
----------------------------------------------
To use VASP with QE, one has to run various calculations in independent directories. Here, we will show examples for the different calculations.

Band Structure
_______________________________________________
1. Create directory called ``scf``.
2. Perform self-consistent calculation in this ``scf`` directory.
3. Create directory called ``bands``.
4. Move the CHGCAR file in the ``scf`` directory to the ``bands`` directory
5. Create a KPOINTS file containing the :math:-`k`-path. This can be done with the :ref:`kpath` feature in PyProcar. 
6. Make sure to set ``LORBIT=11`` or ``LORBIT=12`` for the INCAR in ``bands`` directory
7. Perform a non-self consistent calculation in the ``bands`` by setting ``ICHARG=11`` in the INCAR. 
8. Run pyprocar.bandsplot(dirname = 'bands' ,mode = 'plain', code = 'vasp')

Density of States
_______________________________________________
1. Create directory called ``scf``.
2. Perform self-consistent calculation in this ``scf`` directory.
3. Create directory called ``dos``.
4. Move the CHGCAR file in the ``scf`` directory to the ``dos`` directory.
5. Make sure there is a kmesh in the KPOINTS file in the ``dos`` directory.
6. Make sure to set ``LORBIT=11`` or ``LORBIT=12`` for the INCAR in ``dos`` directory
7. Perform a non-self consistent calculation in the ``dos`` by setting ``ICHARG=11`` in the INCAR. 
8. Run pyprocar.dosplot(dirname = 'bands' ,mode = 'plain', code = 'vasp')

Band Structure and Density of States
_______________________________________________
1. Run the band structure and dos calculation as stated above
2. Run pyprocar.bandsdosplot(bands_dirname = 'bands', dos_dirname = 'dos', bands_mode = 'plain', dos_mode = 'plain', code = 'vasp')

Fermi
_______________________________________________
1. Create directory called ``scf``.
2. Perform self-consistent calculation in this ``scf`` directory.
3. Create directory called ``fermi``.
4. Move the CHGCAR file in the ``scf`` directory to the ``fermi`` directory.
5. Make sure there is a kmesh in the KPOINTS file in the ``fermi`` directory.
6. Make sure to set ``LORBIT=11`` or ``LORBIT=12`` for the INCAR in ``fermi`` directory
7. Perform a non-self consistent calculation in the ``fermi`` by setting ``ICHARG=11`` in the INCAR. 
8. Run pyprocar.FermiHandler(dirname = 'fermi', code = 'vasp')


K-Points Format
_______________________________________________
The :math:`k`-path can be specified in ``KPOINTS`` which is used for the band structure calculation. Here is an example with :math:`k`-path ::


    50 ! Grid points
    Line_mode
    reciprocal
    0.0 0.0 0.0 ! GAMMA
    0.5 -0.5 0.5 ! H

    0.5 -0.5 0.5 ! H
    0.0 0.0 0.5 ! N

    0.0 0.0 0.5 ! N
    0.0 0.0 0.0 ! GAMMA

    0.0 0.0 0.0 ! GAMMA
    0.25 0.25 0.25 ! P

    0.25 0.25 0.25 ! P
    0.5 -0.5 0.5 ! H

    0.25 0.25 0.25 ! P
    0.0 0.0 0.5 ! N



Magnetic Calculations
__________________________________
Magnetic calculations follow the same steps as above, but it requires additional parameters

**Colinear-Spin**
To perform spin colinear calculations set ``ISPIN = 2`` in the ``INCAR``.

**Non-colinear-Spin**
To perform spin non-colinear calculations set ``ISPIN = 2`` and ``LNONCOLLINEAR = .TRUE.``.


