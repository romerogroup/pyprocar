.. _fermi3d:


3D Fermi surface
*************************************

PyProcar's 3D Fermi surface utility is able to generate Fermi surface
plots projected over spin, atoms and orbitals or a combination of one
or many of each. This utility is also capable of projecting external
properties that are provided on a mesh grid in momentum space. This
feature is useful when one wants to project properties that are not
provided in PROCAR file such as Fermi velocity, electron-phonon
coupling and electron effective mass. We divide this section into
three sub sections, plain Fermi surface, projection of properties from
PROCAR and projection of properties from external file.


.. note::
   When plotting 3D Fermi surfaces, it is important to use a
   Monkhorst-Pack k-grid while setting ``ISYM=-1`` in the INCAR file
   (turn off symmetry).
   

3D surfaces
======================================
This section focuses on how the 3d image is created, if you are only
interested in using this functionality, you can skip this part and
move to the `examples <fermi3D.html#id1>`_ section. A surface is
created by adding a collection of small polygons. By increasing the
number of polygons, one can increase the smoothness of the surface. 
These polygons are defined by a collection of points on the surface
also known as vertices and a recipe for their connections to create
the polygons, which is defined by a list of points. This list is
called faces. For example the following defines a cube::

  vertices
   1   1.000000 -1.000000 -1.000000
   2   1.000000 -1.000000  1.000000
   3  -1.000000 -1.000000  1.000000
   4  -1.000000 -1.000000 -1.000000
   5   1.000000  1.000000 -0.999999
   6   0.999999  1.000000  1.000001
   7  -1.000000  1.000000  1.000000
   8  -1.000000  1.000000 -1.000000
  faces
  2 3 4
  8 7 6
  5 6 2
  6 7 3
  3 7 8
  1 4 8
  1 2 4
  5 8 6
  1 5 2
  2 6 3
  4 3 8
  5 1 8

for example the first face defined by a triangle instructs to connect
points 2,3 and 4. One can also assign colors to each face. pyprocar
uses this feature to assign the projection colors to the
surfaces. Another important aspect to this in definning the surface is
the normals to the faces, however a tutorial about fermi surface is
not a good place to get into details about surfaces.

To generate the fermi surface PyProcar uses different levels of
generality to surfaces. The level is the definition of a surface which
is handled by the class surface in `pyprocar.core.Surface
<pyprocar.core.html#module-pyprocar.core.surface>`_
which only requires faces and vertices. The next level is an
isosurface which is handeled by `pyprocar.core.Isosurface
<pyprocar.core.html#module-pyprocar.core.isosurface>`_, this class
represents the equation :math:`f(x,y,z)=V`, where :math:`V` is the
isovalue and :math:`f(x,y,z)` is the function. In a fermi surface
:math:`f(x,y,z)` will be the energy of each band at different points
of the k space and :math:`V` will be the fermi energy. To use this
class one needs to provide the function in a 3 dimentional matrix and
the isovalue. The next level is the fermi surface which is defined by
an isosurfcace which is handled by the class 
`pyprocar.fermisurface3d.FermiSurfcae3D
<pyprocar.fermisurface3d.html#module-pyprocar.fermisurface3d.fermisurface3D>`_.
This function requires a list of kpoints and the eigen-values of the
energy for a specific band. However one does not have to be concerned
about the specifics of different layers and just use the
pyprocar.fermi3D function to generate the fermi surface.
This work would not have been possible without the amazing packages, `pyvista_doc
<https://docs.pyvista.org/>`_ and `trimesh_doc
<https://github.com/mikedh/trimesh>`_.
If you use pyprocar's 3D fermi surface in your publication please cite
trimesh and pyvista as well with the following citation information.

`trimesh_cite <https://github.com/mikedh/trimesh#how-can-i-cite-this-library>`_::

  @software{trimesh,
	author = {{Dawson-Haggerty et al.}},
	title = {trimesh},
	url = {https://trimsh.org/},
	version = {3.2.0},
	date = {2019-12-8},
  }

`PyVista <https://joss.theoj.org/papers/10.21105/joss.01450>`_::

  @article{sullivan2019pyvista,
  doi = {10.21105/joss.01450},
  url = {https://doi.org/10.21105/joss.01450},
  year = {2019},
  month = {may},
  publisher = {The Open Journal},
  volume = {4},
  number = {37},
  pages = {1450},
  author = {C. Bane Sullivan and Alexander Kaszynski},
  title = {{PyVista}: 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit ({VTK})},
  journal = {Journal of Open Source Software}
  }


..  Examples
    ======================================
    As a standard example for plotting fermi surfaces we use MgB\
    :sub:`2`\ and SrFeO\ :sub:`3`\. 


    1. Plain Fermi surface
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    This mode is activated by selecting the argument ``mode`` equal to
    ``'plain'``. This mode plots each band in fermi a different color,
    however there won't be any color projection.

    >>> surfaces = pyprocar.fermi3D(procar='colinear/PROCAR-MgB2-nonPol',
    >>>                            outcar='colinear/OUTCAR-MgB2-nonPol',
    >>>                            mode='plain',
    >>>                            interpolation_factor=4,
    >>>                            projection_accuracy='high',
    >>>                            show=True,
    >>>                            camera_pos=[0, 1, -1],
    >>>                            save2d='MgB2-nonPol.png',
    >>>                            save3d='MgB2-nonPol.glb')


    The main arguments in this function are ``procar``, ``outcar``, ``bands``, ``scale``, ``mode`` and ``st``, where ``procar`` and ``outcar`` are the names of the input PROCAR and OUTCAR files respectively, ``bands`` is an array of the bands that are desired to be plotted. Note if ``bands = -1``, the function will try to plot all the bands provided in the PROCAR file. The kmesh will be interpolated by a factor of scale in each direction. The ``st`` tag controls the spin-texture plotting, and ``mode`` determines the type of projection of colors. There are additional keyword arguments that can be accessed in the help section of this function, such as ``face_color, cmap, atoms, orbitals, energy, transparent, nprocess`` etc. 


.. 2. Surface coloring based on properties from PROCAR
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  Similar to the ``bandsplot()`` section one can choose to project the contribution of different properties provided in the PROCAR file, such as atom, orbital and spin contributions. The projection can be represented by different color mapping schemes chosen by the user. The projection is not restricted to only one property at a time, so it can be chosen from all the provided properties. For example, one might want to see the contribution of the orbitals :math:`p_x`, :math:`p_y`, :math:`p_z` from specific atoms, this function will parse the desired contributions and projects the sum of contributions on each face. To use this functionality one has to change the mode from ``plain`` to ``parametric`` and choose the atoms, orbitals, spin that are desired to be projected.

  For noncolinear calculations, this function is able to plot arrows in the direction of the spinors provided in the PROCAR file. To turn this functionality on the one can set ``st=True`` to turn the spin-texture ON. The user can choose between coloring all the arrows originated from one band with a specific color, or project the contribution of that arrow in a specific Cartesian direction. To better represent the spin-texture we use the key argument ``transparent=True`` which changes the opacity of the Fermi-surface to zero.



    3. Surface coloring based on properties obtained from an external file
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        Similar to the previous section, this function is able to read an external file, containing information about a scalar or a vector field in BZ and project the field on the Fermi surface. This file does not need to have the same mesh grid as the PROCAR file as long as the mesh sampling is fine enough. This function performs an interpolation on the provided data and evaluates functions at the center of each face on the Fermi surface. The external file should have the following format:: 

          band = <band number>
            <kx1>  <ky1>  <kz1>  <color1>
            <kx2>  <ky2>  <kz2>  <color2>
            <kx3>  <ky3>  <kz3>  <color3>
            ...
          band = <band number>
            
        The function matches information about the first band present in the file to the first band requested to be plotted, second band present in the file to the second band requested to be plotted, and so on. 



Keyboard shortcuts
======================================
When plotting with the interactive rendering windows in VTK, several keyboard
shortcuts are available:

+-------------------------------------+-----------------+-----------------------------------------------------+
| Key                                                   | Action                                              |
+=====================================+=================+=====================================================+
| Linux/Windows                       | Mac             |                                                     |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``q``                                                 | Close the rendering window                          |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``f``                                                 | Focus and zoom in on a point                        |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``v``                                                 | Isometric camera view                               |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``w``                                                 | Switch all datasets to a `wireframe` representation |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``r``                                                 | Reset the camera to view all datasets               |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``s``                                                 | Switch all datasets to a `surface` representation   |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``shift+click`` or ``middle-click`` | ``shift+click`` | Pan the rendering scene                             |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``left-click``                      | ``cmd+click``   | Rotate the rendering scene in 3D                    |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``ctrl+click``                      |                 | Rotate the rendering scene in 2D (view-plane)       |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``mouse-wheel`` or ``right-click``  | ``ctl+click``   | Continuously zoom the rendering scene               |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``shift+s``                                           | Save a screenhsot (only on ``BackgroundPlotter``)   |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``shift+c``                                           | Enable interactive cell selection/picking           |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``up``/``down``                                       | Zoom in and out                                     |
+-------------------------------------+-----------------+-----------------------------------------------------+
| ``+``/``-``                                           | Increase/decrease the point size and line widths    |
+-------------------------------------+-----------------+-----------------------------------------------------+

.. automodule:: pyprocar.scripts.scriptFermiHandler
	:members:

