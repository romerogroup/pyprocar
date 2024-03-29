.. _scripts-api-index:

Scripts API
===================================

This section contains functions that users can call to plot and use for pre- or post- processing

PyProcar has the following data types:

- :func:`pyprocar.cat` concatenates PROCAR files

- :func:`pyprocar.bandsplot` plots the band structure.
    The user must provide the directory where the calculation takes place and some case specific files used. 
    Depending on the mode, the user may have to provide more information. 
    For instance is mode='parametric', the user must provide the atoms, orbitald, and spins to include

- :func:`pyprocar.dosplot` plots the density of states.
    The user must provide the directory where the calculation takes place and some case specific files used. 
    Depending on the mode, the user may have to provide more information. 
    For instance is mode='parametric', the user must provide the atoms, orbitald, and spins to include

- :func:`pyprocar.bandsdosplot` plots the density of states and bandtructure in the same figure.
    The is function combines the dosplot and bandsplot. 
    The user is expected to input the bands_setting and dos_setting, which are dictionaries of keyword arguments of dosplot and bandsplot.

- :func:`pyprocar.fermi2d` plots the 2d fermi surface.

- :class:`pyprocar.FermiHandler` plots 3d fermi surface

- :func:`pyprocar.generate2dkmesh` generates a 2d kmesh

- :func:`pyprocar.kpath` generates a K path

- :func:`pyprocar.repair` repairs the PROCAR file

.. toctree::
    :maxdepth: 2

    bandgap
    bandsplot
    bandsdosplot
    cat
    dosplot
    fermi2d
    fermihandler
    generate2dkmesh
    kpath
    repair