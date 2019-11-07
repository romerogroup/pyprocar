2D spin-texture
===============

This module can be utilized to visualize the constant energy surface spin textures in a given system. This feature is particularly useful in identifying Rashba and Dresselhaus type spin-splitting effects, analyzing the topology of Fermi-surface, and examining Lifshitz transitions. To plot 2D spin texture, we require a 2D :math:`k`-grid centered a certain special :math:`k`-point in Brillouin zone near which we want to examine the spin-texture in :math:`k`-space (see section :ref:`labelkmesh` regarding generation of 2D :math:`k`-mesh). 

Usage: To plot :math:`S_x` spin component at a constant energy surface :math:`E = E_{F} + 0.60\,eV` (spin=1, 2, 3 for :math:`S_x`, :math:`S_y`, :math:`S_z`, respectively)::

	pyprocar.fermi2D('PROCAR', outcar='OUTCAR', st=True, energy=0.60, noarrow=True, spin=1)
	 

One could also plot spin texture using arrows instead of a heat map. This can be done by setting the tag: ``noarrow=False``. To set maximum and minimum energy values for color map, one could use ``vmin`` and ``vmax`` tags.

=======================================
Translate and Rotate the 2D KPOINT mesh
=======================================

For any spin-texture in a 2D K-plane, PyProcar usually treats Z-direction as the normal, and makes plot in the X-Y plane. It works fine when we have a 2D k-mesh in (:math:`k_x`, :math:`k_y`, 0) plane, but for other 2D k-meshes (e.g. Y-Z or X-Z), it gives us ``Value Error`` or ``Segmentation Fault``. 

The solution is to rotate the PROCAR by 90 degrees and make the Z-axis perpendicular to the 2D plane. But we should first translate our mesh to a particular K-point (which is mostly the center of the k-mesh) to define the rotation point, and then decide the rotation axis and rotation angle. 

To translate the k-mesh::

	translate=[2,2,0]  

220 is the index of the :math:`k`-point in the 2D :math:`k`-mesh to define the rotation point (counting starts from 0).  

To define the rotation angle and rotation axis::

	rotation=[90,0,1,0]   

This defines the rotation angle = 90, and Y-axis as the rotation axis. PyProcar always performs the translation operation first and then does the rotation.

**Be careful**: Now :math:`S_x`, :math:`S_y` and :math:`S_z` all will be rotated according to the user’s choice. :math:`S_y` will be same as before but :math:`S_x` would now be :math:`S_z` and :math:`S_z` would be :math:`-S_x`.

.. automodule:: pyprocar.scriptFermi2D
	:members: