.. _labeldftprep:

DFT Preparation
================

This section discusses steps to perform DFT calculations to obtain data reqired to run PyProcar for post-processing. Examples of these are available in the ``examples`` directory of the github repository. Features that require non-collinear spin calculations such as 2D spin texture plots and 3D Fermi surfaces with spin texture is currently only supported for VASP calculations. Band unfolding is also limited to VASP calculations since the phase of the wavefunctions is only parsed from the VASP PROCAR file. Support for these features in other DFT codes will be available in the future.

The flag is to be set in PyProcar functions to select the DFT code.

E.g.::

    pyprocar.bandsplot(code='elk', elimit=[-5,5], mode='plain')


========
1. VASP
========

- Required files : PROCAR, OUTCAR (optional), KPOINTS (optional)
- flag           : code='vasp' (default)

In the VASP code, the wavefunction projection information is written into the PROCAR file when ``LORBIT=11`` is set in the INCAR file. For band unfolding, set ``LORBIT=12`` to include phase projections of the wave functions.
An OUTCAR file required to extract the Fermi-energy and reciprocal lattice vectors. If a KPOINTS file is provided, the :math:`k`-path will automatically be labeled in the band structure.
To perform spin colinear calculations set ``ISPIN = 2`` in the ``INCAR``.
To perform spin non-colinear calculatios set ``ISPIN = 2`` and ``LNONCOLLINEAR = .TRUE.``.

First perform a self-consistent calculation with a :math:`k`-mesh grid. Then set ``ICHARG=11`` in the INCAR and create a KPOINTS file containing the :math:-`k`-path. This can be done with the :ref:`labelkpath` feature in PyProcar. 

=======
2. Elk
=======

- Required files : elk.in, BANDLINES.OUT, EFERMI.OUT, LATTICE.OUT, BAND_S*_A*.OUT files
- flag           : code='elk' 

To obtain the required files for Elk, set the following tasks in ``elk.in``::

    tasks
    0
    22

Additionally, for spin colinear calculations set::

    spinpol
    .true.

A :math:`k`-path can be specified in elk.in as follows::

    ! These are the vertices to be joined for the band structure plot
    plot1d
    6 40 
    0.0      0.0      0.0 : \Gamma
    0.5      0.0      0.0 : X
    0.5      0.5      0.0 : M
    0.0      0.0      0.0 : \Gamma
    0.5      0.5      0.5 : R
    0.5      0.0      0.0 : X

First complete the Elk calculation and then run PyProcar in the same directory as the Elk calculations were performed.

===================
3. Quantum Espresso
===================

- Required files : bands.in, kpdos.in, kpdos.out, scf.out
- flag           : code='qe'

Quantum Espresso v6.5+ is supported. 

Run ``pw.x`` for the self-consistent calculation (output : scf.out) and the band structure calculation. The :math:`k`-path can be specified in bands.in which is used for the band structure calculation as follows::

    K_POINTS {crystal_b}
    10
    0.00  0.00  0.00  30 !G
    0.50  0.00  0.00  30 !M
    0.333 0.333 0.00  30 !K
    0.00  0.00  0.00  30 !G
    0.00  0.00  0.50  30 !A
    0.50  0.00  0.50  30 !L
    0.333 0.333 0.50  30 !H
    0.00  0.00  0.50  30 !A|L
    0.50  0.00  0.00  30 !M|K
    0.333 0.333 0.50  30 !H 

Afterwards, to obtain the projections run ``projwfc.x`` on the kpdos.in file to retrieve kpdos.out. PyProcar should be run in this calculation directory.


=========
4. Abinit
=========

- Required files : PROCAR, <abinitoutput>.out
- flag           : code='abinit'

Abinit version :math:`9.x^*` generates a PROCAR similar to that of VASP when ``prtprocar`` is set in the input file. 
To provide the Abinit output file, use the flag ``abinit_output=<nameofoutputfile>`` in PyProcar functions.  

\* Currenlty only available in a development version of Abinit.