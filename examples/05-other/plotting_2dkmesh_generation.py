"""

.. _example_kmesh_generator:

Example of kmesh_generator 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This utility can be used to generate a 2D :math:`k`-mesh centered at a given :math:`k`-point and in a given :math:`k`-plane. 
This is particularly useful in computing 2D spin-textures and plotting 2D Fermi-surfaces. 
For example, the following command creates a 2D :math:`k_{x}`-:math:`k_{y}` -mesh centered at the :math:`\Gamma` point (:math:`k_{z}= 0`) 
ranging from coordinates (-0.5, -0.5, 0.0) to (0.5, 0.5, 0.0) with 5 grids in the x direction and 7 grids in the y direction

.. code-block::
   :caption: General Format

   pyprocar.generate2dkmesh(x1,y1,x2,y2,z,num_grids_x,num_grids_y)

This information is automatically written to a KPOINTS file.
"""
# sphinx_gallery_thumbnail_number = 1

###############################################################################
# Plotting Kmesh
# +++++++++++++++++++++++++++++++++++++++
import pyvista
# You do not need this. This is to ensure an image is rendered off screen when generating exmaple gallery.
pyvista.OFF_SCREEN = True

###############################################################################
# importing pyprocar and specifying local data_dir

import os
import pyprocar

kpoints = pyprocar.generate2dkmesh(-0.5,-0.5,0.5,0.5,0,5,7)

plotter=pyvista.Plotter()
plotter.add_mesh(kpoints, color='blue', render_points_as_spheres=True, point_size=20)
plotter.show_axes()
plotter.show_grid()
plotter.show()
