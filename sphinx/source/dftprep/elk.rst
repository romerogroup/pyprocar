.. _elk:

Elk Perperation
==============================================


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