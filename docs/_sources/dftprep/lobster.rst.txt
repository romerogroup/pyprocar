.. _lobster:

Lobster Perperation
==============================================


- Required files : scf.in, scf.out, lobsterin, lobsterout, FATBAND*.lobter files
- flag           : code='lobster', lobstercode='qe'

Currently supported for Lobster with Quantum Espresso v6.3. 

To use Pyprocar with Lobster, one has to run various calculations in independent directories. Here, we will show examples of the different calculations.

**Band Structure** 

1. Create a directory called ``bands``.
2. Run ``pw.x`` on your ``scf.in`` file. 
3. Run ``lobster.x`` on ``lobsterin``  file in same directory.
4. Run pyprocar.bandsplot(dirname = 'bands' ,mode = 'plain', code = 'qe', lobster = True)


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