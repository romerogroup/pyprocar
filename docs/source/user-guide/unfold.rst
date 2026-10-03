.. _unfold:

Band unfolding
==============

Often times, we need to perform DFT calculations for a supercell geometry rather than the primitive cell. In such cases the band structure becomes quite sophisticated due to the folding of the BZ, and it is difficult to compare the band structure of supercell with that of the primitive cell. The purpose of the band unfolding scheme is to represent the bands within the primitive cell BZ. By calculating the unfolding weight function and plotting the fat bands with the line width proportional to the weight, the unfolded bands can be highlighted. 

Note:
The Brillouin zone of a supercell shrinks respect to the primitive cell. For instance, in a hexagonal primitive lattice the point  H=(1/3, 1/3, 1/2). This point, in a 2x2x2 supercell corresponds to (2/3, 2/3, 1). Therefore, all the distances in the reciprocal space must be doubled (or increased by the respective size of the supercell).


Usage:
First, calculate the band structure of the supercell along the primitive cell's :math:`k`-path, with the :math:`k`-points written in the supercell's reciprocal basis. Set ``LORBIT=12`` in VASP so that the PROCAR includes the phase factors.

Then plot the unfolded bands::

	import numpy as np
	import pyprocar

	fig, ax = pyprocar.unfold(
		code="vasp",
		dirname="supercell",
		mode="plain",
		unfold_mode="both",
		transformation_matrix=np.diag([2, 2, 2]),
		fermi=5.2182,
		elimit=[-5, 5],
		savefig="unfolded_band.png",
	)

``unfold_mode`` sets how the weight is drawn: ``thickness``, ``color`` or ``both``. The ``parametric`` and ``scatter`` modes color the bands by the projection on ``atoms`` and ``orbitals`` and draw the weight as thickness. The overlay modes fill each projection with a thickness of projection times weight. ``unfold`` returns the matplotlib figure and axes, so you can change the plot before you save or show it.

.. automodule:: pyprocar.scripts.scriptUnfold
	:members:
	:undoc-members:
	:inherited-members:
	:show-inheritance: