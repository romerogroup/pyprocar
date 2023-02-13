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
                                code='qe', 
                                spin_calc_type='non-colinear',
                                calc_type='fermi')
"""


###############################################################################
# importing pyprocar and specifying local data_dir
import os
import pyprocar


parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
data_dir = f"{parent_dir}{os.sep}data{os.sep}qe{os.sep}fermi{os.sep}noncolinear{os.sep}Fe"

###############################################################################
# Spin Texture Projection
# +++++++++++++++++++++++++++++++++++++++
#
# By setting spin_texture to be true, You can plot the arrows for the spin textures.
# By default the projected values of the arrows will be s_z. 
# But you can change this by setting arrow_projection to one of the following
# 'x','y','z','x^2','y^2','z^2'


pyprocar.fermi2D(code = 'qe',
               dirname=data_dir,
               spin_texture=True,
               arrow_projection='x',
               arrow_size =15,
               arrow_density=10,
               color_bar=True)


###############################################################################
# Spin Texture single color
# +++++++++++++++++++++++++++++++++++++++
#


pyprocar.fermi2D(code = 'qe',
               dirname=data_dir,
               spin_texture=True,
               arrow_color = 'blue',
               arrow_size =15,
               arrow_density=10)

