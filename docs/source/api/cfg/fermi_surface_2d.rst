fermi surface 2d plotting Options
=====================================================

:func:`pyprocar.fermi2D` takes its plotting options as keyword arguments.

mode
----

:description: ``plain``, ``plain_bands``, ``parametric`` or ``spin_texture``. ``parametric`` colors the contours by the ``atoms``, ``orbitals`` and ``spins`` projection. ``spin_texture`` needs a non-collinear calculation.

:value: plain


k_z_plane
---------

:description: The k_z coordinate of the slicing plane

:value: 0.0


extend_zone_directions
----------------------

:description: Directions to extend the surface into neighboring Brillouin zones

:value: None


show_colorbar
-------------

:description: Boolean to draw a colorbar in ``parametric`` and ``spin_texture`` modes

:value: True


cmap
----

:description: The colormap for the projection colors and spin arrows

:value: plasma


plot_line_kwargs
----------------

:description: Keyword arguments for the matplotlib LineCollection of the contours, such as ``{"colors": "purple", "linewidths": 2.0, "linestyles": "dashed"}``

:value: None


plot_arrows
-----------

:description: Boolean to draw spin arrows in ``spin_texture`` mode

:value: True


plot_arrows_kwargs
------------------

:description: Keyword arguments for the matplotlib quiver of the spin arrows, such as ``{"scale": 2.0}``

:value: None


padding
-------

:description: Padding of the k-mesh for the Fermi surface calculation

:value: 10


figsize
-------

:description: The figure size in inches

:value: (8, 6)


dpi
---

:description: The figure resolution

:value: 100
