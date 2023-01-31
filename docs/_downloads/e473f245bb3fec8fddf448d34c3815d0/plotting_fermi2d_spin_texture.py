"""

.. _ref_plotting_fermi2d_noncolinear:

Plotting fermi2d noncolinear
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
# Spin Texture
# +++++++++++++++
#
#

pyprocar.fermi2D(code = 'qe',
               dirname=data_dir,
               spin_texture=True)

