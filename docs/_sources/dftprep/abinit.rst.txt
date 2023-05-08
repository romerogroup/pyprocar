.. _abinit:

Abinit Perperation
==============================================


- Required files : PROCAR, <abinitoutput>.out
- flag           : code='abinit'

Abinit version :math:`9.x^*` generates a PROCAR similar to that of VASP when ``prtprocar`` is set in the input file. 
To provide the Abinit output file, use the flag ``abinit_output=<nameofoutputfile>`` in PyProcar functions.  

When running Abinit in parallel the PROCAR is split into multiple files. PyProcar's ``cat`` function can merge these files together as explained in the section :ref:`cat`. 
This also has an option to fix formatting issues in the Abinit PROCAR if needed. 

