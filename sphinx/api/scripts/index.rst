.. _script-api-index:

Scripts API
========

This section contains functions that users can call to plot and use for pre- or post- processing

PyProcar has the following data types:

- :function:`pyprocar.bandsplot` plot the band structure.
The user must provide the directory where the calculation takes place and some case specific files used. 
Depending on the mode, the user may have to provide more information. 
For instance is mode='parametric', the user must provide the atoms, orbitald, and spins to include

- :function:`pyprocar.dosplot` plot the density of states.
The user must provide the directory where the calculation takes place and some case specific files used. 
Depending on the mode, the user may have to provide more information. 
For instance is mode='parametric', the user must provide the atoms, orbitald, and spins to include

- :function:`pyprocar.bandsdosplot` plots the density of states and bandtructure in the same figure.
The is function combines the dosplot and bandsplot. 
The user is expected to input the bands_setting and dos_setting, which are dictionaries of keyword arguments of dosplot and bandsplot.
