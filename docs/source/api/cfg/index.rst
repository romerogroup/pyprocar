.. _cfg-index:

Configuration
=============

Each plotting function reads its default options from a configuration class in ``pyprocar/cfg/``.
Pass an option as a keyword argument to override it, for example
``pyprocar.bandsplot(..., fermi_color="red")``. Set ``print_plot_opts=True`` to print the options
and their current values.

- ``pyprocar/cfg/band_structure.py`` : ``BandStructureConfig``, the options of ``pyprocar.bandsplot``

- ``pyprocar/cfg/band_structure_2d.py`` : ``Bandstructure2DConfig``, the options of ``pyprocar.BandStructure2DHandler``

- ``pyprocar/cfg/dos.py`` : ``DensityOfStatesConfig``, the options of ``pyprocar.dosplot``

- ``pyprocar/cfg/unfold.py`` : ``UnfoldingConfig``, the options of ``pyprocar.unfold``

- ``pyprocar/cfg/fermi_surface_3d.py`` : ``FermiSurface3DConfig``, the options of ``pyprocar.FermiHandler``

- ``fermi_surface_2d`` : The plotting options of ``pyprocar.fermi2D``, which takes them as keyword arguments


.. toctree::
   :maxdepth: 1

   band_structure
   band_structure_2d
   dos
   unfold
   fermi_surface_3d
   fermi_surface_2d
