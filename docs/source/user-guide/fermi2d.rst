.. _fermi2d:

2D spin-texture
===============

This module can be utilized to visualize the constant energy surface spin textures in a given system. This feature is particularly useful in identifying Rashba and Dresselhaus type spin-splitting effects, analyzing the topology of Fermi-surface, and examining Lifshitz transitions. To plot 2D spin texture, we require a 2D :math:`k`-grid centered a certain special :math:`k`-point in Brillouin zone near which we want to examine the spin-texture in :math:`k`-space (see section :ref:`example_kmesh_generator` regarding generation of 2D :math:`k`-mesh). 

Usage: To plot the :math:`S_x` spin component at a constant energy surface :math:`E = E_{F} + 0.60\,eV` (spins=[1], [2], [3] for :math:`S_x`, :math:`S_y`, :math:`S_z`, respectively)::

	pyprocar.fermi2D(code='vasp', dirname='fermi', mode='parametric', energy=0.60, spins=[1])

One could also plot the spin texture with arrows instead of a heat map by setting ``mode='spin_texture'``. The arrows show the in-plane spin direction and their color shows the spin magnitude. To change the colormap, use ``cmap``.

======================================
Customize the plot with matplotlib
======================================

``fermi2D`` returns the matplotlib figure and axes, so the plot can be processed further with any matplotlib option.
Usage::

    import matplotlib.pyplot as plt
    import pyprocar

    fig, ax = pyprocar.fermi2D(code='vasp', dirname='fermi', show=False)
    ax.set_title('Using matplotlib options')
    plt.show()


.. automodule:: pyprocar.scripts.scriptFermi2D
	:members:
