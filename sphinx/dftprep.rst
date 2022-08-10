.. _labeldftprep:

DFT Preparation
================

This section discusses steps to perform DFT calculations to obtain data required to run PyProcar for post-processing. Examples of these are available in the ``examples`` directory of the Github repository. Features that require non-collinear spin calculations such as 2D spin texture plots and 3D Fermi surfaces with spin texture is currently only supported for VASP calculations. Band unfolding is also limited to VASP calculations since the phase of the wavefunctions is only parsed from the VASP PROCAR file. Support for these features in other DFT codes will be available in the future.

The flag is to be set in PyProcar functions to select the DFT code.

E.g.::

    pyprocar.bandsplot(code='elk', elimit=[-5,5], mode='plain')


========
1. VASP
========

- Required files : PROCAR, OUTCAR (optional), KPOINTS (optional)
- flag           : code='vasp' (default)

In the VASP code, the wavefunction projection information is written into the PROCAR file when ``LORBIT=11`` is set in the INCAR file. For band unfolding, set ``LORBIT=12`` to include phase projections of the wave functions.
An OUTCAR file is required to extract the Fermi-energy and reciprocal lattice vectors. If a KPOINTS file is provided, the :math:`k`-path will automatically be labeled in the band structure.
To perform spin colinear calculations set ``ISPIN = 2`` in the ``INCAR``.
To perform spin non-colinear calculations set ``ISPIN = 2`` and ``LNONCOLLINEAR = .TRUE.``.

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

First, complete the Elk calculation and then run PyProcar in the same directory as the Elk calculations were performed.

===================
3. Quantum Espresso
===================

- Required files : bands.in, kpdos.in, pdos.in,scf.in, atomic_proj.xml
- flag           : code='qe'

Quantum Espresso v6.5+ is supported. 

To use Pyprocar with QE, one has to run various calculations in independent directories. Here, we will show examples for the different calculations

**Band Structure** 

1. Create directory called ``bands``.
2. Run ``pw.x`` on your ``scf.in`` file. 
3. Run ``pw.x`` on your ``bands.in`` file.  (Read kpoints section on how to add labels)
4. Run ``projwfc.x`` on your ``kpdos.in`` file (Make sure kresolveddos=.true.). 
5. Make sure to copy the atomic_proj.xml file that is found in the .save directory into the main directory
6. Run pyprocar.bandsplot(dirname = 'bands' ,mode = 'plain', code = 'qe')

**Density of States** 

1. Create directory called ``dos``. 
2. Run ``pw.x`` on your ``scf.in`` file. 
3. Run ``pw.x`` on your ``nscf.in`` file. 
4. Run ``projwfc.x`` on your ``pdos.in`` file (Make sure kresolveddos=.false.). 
5. Make sure to copy the atomic_proj.xml file that is found in the .save directory into the main directory
6. Run pyprocar.dosplot(dirname = 'bands' ,mode = 'plain', code = 'qe')

**Band Structure and Density of States** 

1. Run the band structure and dos calculation as stated above

5. Run pyprocar.bandsdosplot(bands_dirname = 'bands', dos_dirname = 'dos', bands_mode = 'plain', dos_mode = 'plain', code = 'qe')

**Fermi** 

1. Create directory called ``fermi``. 
2. Run ``pw.x`` on your ``scf.in`` file. 
3. Run ``pw.x`` on your ``nscf.in`` file. 
4. Run ``projwfc.x`` on your ``kpdos.in`` file (Make sure kresolveddos=.true.). 
5. Make sure to copy the atomic_proj.xml file that is found in the .save directory into the main directory
6. Run pyprocar.fermi3D


**K-Points Format**

The :math:`k`-path can be specified in ``bands.in`` which is used for the band structure calculation as one of the following::


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


Where the one occurs is at the place of a discontinuity.

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

============
4. Lobster
============

- Required files : scf.in, scf.out, lobsterin, lobsterout, FATBAND*.lobter files
- flag           : code='lobster', lobstercode='qe'

Currently supported for Lobster with Quantum Espresso v6.3. 

To use Pyprocar with Lobster, one has to run various calculations in independent directories. Here, we will show examples of the different calculations.

**Band Structure** 

1. Create a directory called ``bands``.
2. Run ``pw.x`` on your ``scf.in`` file. 
3. Run ``lobster.x`` on ``lobsterin``  file in same directory.
6. Run pyprocar.bandsplot(dirname = 'bands' ,mode = 'plain', code = 'qe', lobster = True)

**Density of States** 

1. Create a directory called ``dos``.
2. Run ``pw.x`` on your ``scf.in`` file. 
3. Run ``lobster.x`` on ``lobsterin``  file in same directory.
4. Run pyprocar.dosplot(dirname = 'dos' ,mode = 'plain', code = 'qe', lobster = True)

**Band Structure and Density of States** 

1. Run the band structure and dos calculation as stated above

5. Run pyprocar.bandsdosplot(bands_dirname = 'bands', dos_dirname = 'dos', bands_mode = 'plain', dos_mode = 'plain', code = 'qe', lobster = True)

**Fermi** 

1. Create a directory called ``fermi``.
2. Run ``pw.x`` on your ``scf.in`` file. 
3. Run ``lobster.x`` on ``lobsterin`` file in same directory.
4. Run pyprocar.fermi3D

**KPOINTS**

The kpoints for a lobster calculation must be listed in a specific format for a particular DFT code. Right now we only support QE, but additional support will be added for VASP and ABINIT.

**QE**

You must have the following settings for lobster:

-  wf_collect = .true. in CONTROL
-  nosym = .TRUE., noinv = .TRUE. in SYSTEM::


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

lobster_input_file must include explicit bands such as::


    createFatband F 2p_x 2p_y 2p_z 2s
    createFatband Li 1s 2s

Refer to Fe/Lobster_QE in the examples directory for example inputs

=========
5. Abinit
=========

- Required files : PROCAR, <abinitoutput>.out
- flag           : code='abinit'

Abinit version :math:`9.x^*` generates a PROCAR similar to that of VASP when ``prtprocar`` is set in the input file. 
To provide the Abinit output file, use the flag ``abinit_output=<nameofoutputfile>`` in PyProcar functions.  

When running Abinit in parallel the PROCAR is split into multiple files. PyProcar's ``cat`` function can merge these files together as explained in the section :ref:`labelcat`. This also has an option to fix formatting issues in the Abinit PROCAR if needed. 

