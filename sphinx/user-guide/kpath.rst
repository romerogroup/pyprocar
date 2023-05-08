.. _kpath:

K Path
===============

PyProcar provides a data class to manage the kpath information when performing band structure calculations, 
known as the KPath class. This class takes knames, kticks, special_kpoints, and ngrids as arguments


Accessing Kpath Information
+++++++++++++++++++++++++++++++++++++

The KPath object (referred to as "kpath") can be accessed through the main io.Parser class 
or the ElectronicBandStructure:

.. code-block:: python

    import pyprocar

    parser = pyprocar.io.Parser(code = 'vasp', dir=path_to_calculation)

    kpath = parser.kpath

    ebs = parser.ebs
    kpath = ebs.kpath


Using the kapth object, you can access various information related to the kpath:

.. code-block:: python

    kpath.nsegments # The number of kpath segments
    # 6

    kpath.knames # The knames
    # [['$GAMMA$', '$H$'], ['$H$', '$N$'], ['$N$', '$GAMMA$'], ['$GAMMA$', '$P$'], ['$P$', '$H$'], ['$P$', '$N$']]

    kpath.tick_positions # The ticks which the knames belong to
    # [0, 49, 99, 149, 199, 249, 299]

    kpath.tick_names # The knames
    # ['$GAMMA$', '$H$', '$N$', '$GAMMA$', '$P$', '$H$|$P$', '$N$']

    kpath.kdistances # The distances along the kpath 
    # [0.8660254  0.70710678 0.5        0.4330127  0.8291562  0.4330127 ]