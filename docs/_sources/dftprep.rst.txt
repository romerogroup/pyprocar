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

Run ``pw.x`` for the self-consistent calculation (output : scf.out) and the band structure calculation. The :math:`k`-path can be specified in bands.in which is used for the band structure calculation as one of the following::


    K_POINTS {crystal_b}
    8
        0.0000000000       0.0000000000       0.0000000000       30 !G
        0.5000000000       0.0000000000       0.5000000000       30 !X
        0.6250000000       0.2500000000       0.6250000000       1  !U
        0.3750000000       0.3750000000       0.7500000000       30 !K
        0.0000000000       0.0000000000       0.0000000000       30 !G
        0.5000000000       0.5000000000       0.5000000000       30 !L
        0.5000000000       0.2500000000       0.7500000000       30 !W
        0.5000000000       0.0000000000       0.5000000000       30 !X


The on labels a discontinuity that occurs.

Explicit::


    K_POINTS {crystal}
    269
        0.0000000000       0.0000000000       0.0000000000      1.0 !G
        0.0083333333       0.0000000000       0.0083333333      1.0
        0.0166666667       0.0000000000       0.0166666667      1.0
        0.0250000000       0.0000000000       0.0250000000      1.0
        0.0333333333       0.0000000000       0.0333333333      1.0
        0.0416666667       0.0000000000       0.0416666667      1.0
        .
        .
        .
        0.4916666667       0.0000000000       0.4916666667      1.0
        0.5000000000       0.0000000000       0.5000000000      1.0 !X
        0.5062500000       0.0125000000       0.5062500000      1.0 
        .
        .
        .
        0.6125000000       0.2250000000       0.6125000000      1.0
        0.6187500000       0.2375000000       0.6187500000      1.0
        0.6250000000       0.2500000000       0.6250000000      1.0 !U
        0.3750000000       0.3750000000       0.7500000000      1.0 !K
        0.3691406250       0.3691406250       0.7382812500      1.0
        0.3632812500       0.3632812500       0.7265625000      1.0
        0.3574218750       0.3574218750       0.7148437500      1.0
        .
        .
        .
        0.0058593750       0.0058593750       0.0117187500      1.0
        0.0000000000       0.0000000000       0.0000000000      1.0 !G


- Explicitly listing kpoints as ''!kpoint" is important for labels

To perform spincalcs set nspin = 2 and starting_magnetization(1)= 0.7

lobster_input_file must include explicit bands such as::


    createFatband F 2p_x 2p_y 2p_z 2s
    createFatband Li 1s 2s

Afterwards, to obtain the projections run ``projwfc.x`` on the kpdos.in file to retrieve kpdos.out. PyProcar should be run in this calculation directory.

============
4. Lobster
============

- Required files : scf.in, scf.out, lobsterin, lobsterout, FATBAND*.lobter files
- flag           : code='lobster', lobstercode='qe'

Currently supported for Lobster with Quantum Espresso v6.3. 

You must have the following settings for lobster:

-  wf_collect = .true. in CONTROL

-   nosym = .TRUE., noinv = .TRUE. in SYSTEM

The kpoints for a lobster file must be listed in the scf.in file as the following::


    K_POINTS crystal
    520
    0.0000000   0.0000000   0.0000000   1.0
    0.0000000   0.0000000   0.1428571   1.0
    0.0000000   0.0000000   0.2857143   1.0
    0.0000000   0.0000000   0.4285714   1.0
    .
    .
    .
    -0.1428571  -0.1428571  -0.2857143   1.0
    -0.1428571  -0.1428571  -0.1428571   1.0
    0.0000000000     0.0000000000     0.0000000000 0.0000 !G
    0.0200000000     0.0200000000     0.0200000000 0.0000
    .
    .
    .
    0.4800000000     0.4800000000     0.4800000000 0.0000
    0.5000000000     0.5000000000     0.5000000000 0.0000 !T
    0.5110420726     0.4889579274     0.5000000000 0.0000
    .
    .
    .
    0.7539676705     0.2460323295     0.5000000000 0.0000
    0.7650097432     0.2349902568     0.5000000000 0.0000 !H2
    0.5000000000    -0.2349902568     0.2349902568 0.0000 !H0
    0.5000000000    -0.2238002446     0.2238002446 0.0000
    .
    .


- The k meth and kpath must be listed explicitly. kmesh gets a weight of 1, and the path gets a weight of 0.
- Explicitly listing kpoints as ''!kpoint" on the k path is important for labels

To perform spincalcs set nspin = 2 and starting_magnetization(1)= 0.7

Follow instructions on how to perform a Lobster analysis with Quantum Espresso. Also refer to the files in the relevant ``examples`` directory.

=========
5. Abinit
=========

- Required files : PROCAR, <abinitoutput>.out
- flag           : code='abinit'

Abinit version :math:`9.x^*` generates a PROCAR similar to that of VASP when ``prtprocar`` is set in the input file. 
To provide the Abinit output file, use the flag ``abinit_output=<nameofoutputfile>`` in PyProcar functions.  

When running Abinit in parallel the PROCAR is split into multiple files. PyProcar's ``cat`` function can merge these files together as explained in the section :ref:`labelcat`. This also has an option to fix formatting issues in the Abinit PROCAR if needed. 

