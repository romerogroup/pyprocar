3D Fermi surface
================

PyProcar's 3D Fermi surface utility is able to generate Fermi surface plots projected over spin, atoms and orbitals or a combination of one or many of each. This utility is also capable of projecting external properties that are provided on a mesh grid in momentum space. This feature is useful when one wants to project properties that are not provided in PROCAR file such as Fermi velocity, electron-phonon coupling and electron effective mass. We divide this section into three sub sections, plain Fermi surface, projection of properties from PROCAR and projection of properties from external file. 

======================
1. Plain Fermi surface
======================

Usage::

	pyprocar.fermi3D(procar,outcar,bands,scale=1,mode='plain',st=False,**kwargs)

The main arguments in this function are ``procar``, ``outcar``, ``bands``, ``scale``, ``mode`` and ``st``, where ``procar`` and ``outcar`` are the names of the input PROCAR and OUTCAR files respectively, ``bands`` is an array of the bands that are desired to be plotted. Note if ``bands = -1``, the function will try to plot all the bands provided in the PROCAR file. The kmesh will be interpolated by a factor of scale in each direction. The ``st`` tag controls the spin-texture plotting, and ``mode`` determines the type of projection of colors. There are additional keyword arguments that can be accessed in the help section of this function, such as ``face_color, cmap, atoms, orbitals, energy, transparent, nprocess`` etc. 

===================================================
2. Surface coloring based on properties from PROCAR
===================================================

Similar to the ``bandsplot()`` section one can choose to project the contribution of different properties provided in the PROCAR file, such as atom, orbital and spin contributions. The projection can be represented by different color mapping schemes chosen by the user. The projection is not restricted to only one property at a time, so it can be chosen from all the provided properties. For example, one might want to see the contribution of the orbitals :math:`p_x`, :math:`p_y`, :math:`p_z` from specific atoms, this function will parse the desired contributions and projects the sum of contributions on each face. To use this functionality one has to change the mode from ``plain`` to ``parametric`` and choose the atoms, orbitals, spin that are desired to be projected.

For noncolinear calculations, this function is able to plot arrows in the direction of the spinors provided in the PROCAR file. To turn this functionality on the one can set ``st=True`` to turn the spin-texture ON. The user can choose between coloring all the arrows originated from one band with a specific color, or project the contribution of that arrow in a specific Cartesian direction. To better represent the spin-texture we use the key argument ``transparent=True`` which changes the opacity of the Fermi-surface to zero.

======================================================================
3. Surface coloring based on properties obtained from an external file
======================================================================

Similar to the previous section, this function is able to read an external file, containing information about a scalar or a vector field in BZ and project the field on the Fermi surface. This file does not need to have the same mesh grid as the PROCAR file as long as the mesh sampling is fine enough. This function performs an interpolation on the provided data and evaluates functions at the center of each face on the Fermi surface. The external file should have the following format:: 

	band = <band number>
	   <kx1>  <ky1>  <kz1>  <color1>
	   <kx2>  <ky2>  <kz2>  <color2>
	   <kx3>  <ky3>  <kz3>  <color3>
	   ...
	band = <band number>
    
The function matches information about the first band present in the file to the first band requested to be plotted, second band present in the file to the second band requested to be plotted, and so on. 

.. automodule:: pyprocar.scriptFermi3D
	:members: