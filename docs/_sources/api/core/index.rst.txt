.. _core-api-index:

Core API
========

This chapter is intended to describe data types that are used to assist in the processing of the electronic structure

PyProcar has the following data types:

- :class:`pyprocar.BrillouinZone` is used to generates the Brillouin Zone from reciprocal lattice matrix. 

- :class:`pyprocar.FermiSurface3D` is used to store the informtation of a 3D Fermi Surface. This includes the interpolated kpoints, isosurface, and scalar value 
This class is an extension of the general :class:`pyprocar.Isosurface` 

- :class:`pyprocar.Isosurface` is used to generate an isosurface for an isovalue
from a numpy array of coordinate points and a a 1d numpy array of scalar values. 
This class is an extension of the :class:`pyprocar.Surface class`

- :class:`pyprocar.Surface` is used to generate a surface from vertices and faces.
This class is an extension of the :class:`pyvista.PolyData class`

- :class:`pyprocar.KPath` is used to store the k-path information (tick_labels,tick_positions,n_segements,kdistances) 
for band structure plots

- :class:`pyprocar.ElectronicBandStructure` is used to store and manipulate electronic band structure information. 
Electronic band structures expects kpoints, band values, fermi energy, projections, phase projections, kpath, weights, and reciprocal lattice.

- :class:`pyprocar.DensityOfStates` is used to store and manipulate density of states information.
Density of states 1d numpy array of energy, (n_energy,n_spin) total densities and projections at each energy.

- :class:`pyprocar.FermiSurface` is used to help plot the 2d fermi surface at a given plane.
Fermi Surface expects numpy array of k points, band energies, and projections.

- :class:`pyprocar.Structure` is used to store structure information. 
Structure expects a list of atomic symbols, numpy array of fraction coordinates of the atoms, and the atomic lattice matrix.