"""

.. _ref_plotting_fermi2d_noncolinear:

Plotting fermi2d noncolinear
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting fermi2d noncolinear example. For more information about fermi2d refer to :ref:`fermi2d`

First download the example files with the code below. Then replace data_dir below.

.. code-block::
   :caption: Downloading example

    data_dir = pyprocar.download_example(save_dir='', 
                                material='Fe',
                                code='vasp', 
                                spin_calc_type='non-colinear',
                                calc_type='fermi')
"""

###############################################################################
# importing pyprocar and specifying local data_dir
import os

import pyprocar

data_dir = os.path.join(
    pyprocar.utils.DATA_DIR, "examples", "Fe", "vasp", "non-colinear", "fermi"
)

###############################################################################
# Spin Texture Projection
# +++++++++++++++++++++++++++++++++++++++
#
# By setting spin_texture to be true, You can plot the arrows for the spin textures.
# By default the projected values of the arrows will be s_z.
# But you can change this by setting arrow_projection to one of the following
# 'x','y','z','x^2','y^2','z^2'


pyprocar.fermi2D(
    code="vasp",
    dirname=data_dir,
    fermi=5.5962,
    spin_texture=True,
    spin_projection="x",
    arrow_size=3,
    arrow_density=10,
    plot_color_bar=True,
)


###############################################################################
# Spin Texture single color
# +++++++++++++++++++++++++++++++++++++++
#


pyprocar.fermi2D(
    code="vasp",
    dirname=data_dir,
    fermi=5.5962,
    spin_texture=True,
    arrow_color="blue",
    arrow_size=3,
    arrow_density=10,
)


###############################################################################
# Selecting band indices
# +++++++++++++++++++++++++++++++++++++++
#
# You can specify specfic bands with the band indices keyword.
# band_indices will be a list of list that contain band indices. Below I only plot bands 14,15
# Also you can specify the colors of the bands as well with band_colors

band_indices = [[6, 7]]
band_colors = [["blue", "red"]]
pyprocar.fermi2D(
    code="vasp",
    mode="plain_bands",
    fermi=5.5962,
    band_indices=band_indices,
    band_colors=band_colors,
    spin_texture=True,
    arrow_size=3,
    arrow_density=10,
    dirname=data_dir,
)
