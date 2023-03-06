.. _qe: 

Quantum Espresso Preparation
==============================================

This guide is here to help you prepare Quantum Espresso Calculations to be used with pyprocar.


- Required files : bands.in, kpdos.in, pdos.in,scf.in, atomic_proj.xml
- flag           : code='qe'

Quantum Espresso v6.5+ is supported. 


Preparing Calculations
----------------------------------------------
To use Pyprocar with QE, one has to run various calculations in independent directories. Here, we will show examples for the different calculations


Band Structure
_______________________________________________
1. Create directory called ``bands``.
2. Run ``pw.x`` on your ``scf.in`` file. 
3. Run ``pw.x`` on your ``bands.in`` file.  (Read kpoints section on how to add labels)
4. Run ``projwfc.x`` on your ``kpdos.in`` file (Make sure kresolveddos=.true.). 
5. Make sure to copy the atomic_proj.xml file that is found in the .save directory into the main directory
6. Run pyprocar.bandsplot(dirname = 'bands' ,mode = 'plain', code = 'qe')

Density of States
_______________________________________________
1. Create directory called ``dos``. 
2. Run ``pw.x`` on your ``scf.in`` file. 
3. Run ``pw.x`` on your ``nscf.in`` file. 
4. Run ``projwfc.x`` on your ``pdos.in`` file (Make sure kresolveddos=.false.). 
5. Make sure to copy the atomic_proj.xml file that is found in the .save directory into the main directory
6. Run pyprocar.dosplot(dirname = 'bands' ,mode = 'plain', code = 'qe')

Band Structure and Density of States 
_______________________________________________
1. Run the band structure and dos calculation as stated above
2. Run pyprocar.bandsdosplot(bands_dirname = 'bands', dos_dirname = 'dos', bands_mode = 'plain', dos_mode = 'plain', code = 'qe')

Fermi
_______________________________________________
1. Create directory called ``fermi``. 
2. Run ``pw.x`` on your ``scf.in`` file. 
3. Run ``pw.x`` on your ``nscf.in`` file. 
4. Run ``projwfc.x`` on your ``kpdos.in`` file (Make sure kresolveddos=.true.). 
5. Make sure to copy the atomic_proj.xml file that is found in the .save directory into the main directory
6. Run pyprocar.fermi3D


K-Points Format
_______________________________________________
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



Magnetic Calculations
__________________________________
Magnetic calculations follow the same steps as above, but it requires additional parameters



**Colinear-Spin**

To perform Colinear-Spin calculations, nspin = 2 and starting_magnetization must be set in the input of the PW files (scf.in,nscf.in,bands.in,)



**Non-colinear-Spin**

Non-colinear-Spin-Spin calculations take some additional steps as it requires a branch of Quantum Espresso code to print out the spin projections.

Follow these steps to install the qe branch:

1. git clone git@gitlab.com:pietrodelugas/q-e.git
2. cd q-e
3. git checkout new_proj
4. Install package. 
5. Set PATH to the bin directory in side q-e


Now, to perform the calculations set noncolin = .true. and lspinorb = .true. in the input of the PW input files (scf.in,nscf.in,bands.in,). 

Also, set savesigma=.true. in the PROJWFC input files (kpdos.in,pdos.in). 
