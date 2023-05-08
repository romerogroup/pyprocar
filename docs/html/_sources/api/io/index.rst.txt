.. _io-api-index:

IO API
===================================

This section decribes the input and output of PyProcar. This handles the parsing of the density functional codes

PyProcar has the following parsers

- :class:`pyprocar.io.qe.QEParser` is used to parse Quantum Espresso results. 

- :class:`pyprocar.io.bxsf.BxsfParser` is used to parse Bxsf formatted results. 

- :class:`pyprocar.io.lobster.LobsterParser` is used to parse Lobster results. 

- :class:`pyprocar.io.siesta.SiestaParser` is used to parse Siesta results. 

- :mod:`pyprocar.io.vasp` This module is used to parse Vasp results

- :mod:`pyprocar.io.abinit` This module is used to parse Abinit results

.. toctree::
    :maxdepth: 2

    abinit
    bxsf
    lobster
    qe
    siesta
    vasp
   
