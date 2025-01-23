"""

.. _ref_plotting_fermi3d_isovalue_gif:

Plotting fermi3d isovalue_gif
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symmetry does not currently work! Make sure for fermi surface calculations turn off symmetry

Plotting fermi3d isovalue_gif example.

First download the example files with the code below. Then replace data_dir below.

.. code-block::
   :caption: Downloading example

    data_dir = pyprocar.download_example(save_dir='', 
                                material='Fe',
                                code='qe', 
                                spin_calc_type='non-spin-polarized',
                                calc_type='fermi')
"""

# sphinx_gallery_thumbnail_number = 1

###############################################################################

import pyvista

# You do not need this. This is to ensure an image is rendered off screen when generating exmaple gallery.
pyvista.OFF_SCREEN = True

###############################################################################
# importing pyprocar and specifying local data_dir

import os

import pyprocar

data_dir = os.path.join(
    pyprocar.utils.DATA_DIR, "examples", "Fe", "qe", "non-spin-polarized", "fermi"
)

# First create the FermiHandler object, this loads the data into memory. Then you can call class methods to plot
# Symmetry only works for specfic space groups currently.
# For the actual calculations turn off symmetry and set 'apply_symmetry'=False
fermiHandler = pyprocar.FermiHandler(code="qe", dirname=data_dir, apply_symmetry=True)

###############################################################################
# Plain mode
# +++++++++++++++++++++++++++++++++++++++
#
#


# iso_range will be the energy range around the fermi level. 2 would search 1 ev above and below.
iso_range = 2

# iso_surface will generate 5 surfaces equally space throughout the range.
iso_surfaces = 5

# Instead of iso_range and iso_surfaces, you can specify exact energy values to generate isosurfaces
iso_values = [-1, -0.5, 0.25, 1, 5]

fermiHandler.create_isovalue_gif(
    iso_range=iso_range,
    iso_surfaces=iso_surfaces,
    save_gif="isovalue_gif.gif",
    mode="plain",
)
